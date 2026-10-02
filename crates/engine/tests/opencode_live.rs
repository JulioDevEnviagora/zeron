//! Opt-in: real engine turns against the user's real opencode (PATH or
//! OPENCODE_EXECUTABLE), with the Zeron MCP injection every app run gets.
//! OC_MODEL picks `provider/model` (default opencode/big-pickle); OC_MCP is
//! the MCP server command (default: this test binary, which exits at once).
//!
//!     cargo test -p zeron-engine --test opencode_live -- --ignored --nocapture
use std::{sync::Arc, time::Duration};
use zeron_doc::{MessagePart, MessageRole, MessageStatus, SessionCommandPayload};
use zeron_engine::{EngineCore, HarnessRegistry};
use zeron_harness::OpencodeHarness;
use zeron_proto::{HarnessId, RunRequest, SandboxLevel, SessionStatus};

const CHAT: &str = "opencode-live";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "drives a real opencode install"]
async fn real_opencode_answers_two_turns() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    let cwd = project.to_str().unwrap().to_owned();
    let registry = HarnessRegistry::new();
    registry.register(Arc::new(OpencodeHarness::new()));
    let core = EngineCore::assemble(
        &dir.path().join("engine"),
        Arc::new(registry),
        HarnessId::Opencode,
        None,
    )
    .unwrap();
    core.workspace
        .create_space("p", &core.device_id, &cwd, None, false)
        .unwrap();
    core.workspace
        .create_chat(CHAT, Some("p"), None, None, None)
        .unwrap();
    let model = std::env::var("OC_MODEL").unwrap_or_else(|_| "opencode/big-pickle".into());
    let mcp_command = std::env::var("OC_MCP").unwrap_or_else(|_| {
        std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    });
    let mut failures = Vec::new();
    for turn in 0..2 {
        let started = std::time::Instant::now();
        core.doc_host
            .queue_command(
                CHAT,
                SessionCommandPayload::Run {
                    message_id: uuid::Uuid::new_v4().to_string(),
                    request: RunRequest {
                        mcp: Some(zeron_proto::McpServer {
                            name: "zeron".into(),
                            command: mcp_command.clone(),
                            args: vec!["mcp".into()],
                            env: Default::default(),
                        }),
                        prompt: "Reply with exactly: PONG".into(),
                        harness: Some(HarnessId::Opencode),
                        model: Some(model.clone()),
                        reasoning: None,
                        model_options: Default::default(),
                        cwd: cwd.clone(),
                        sandbox: SandboxLevel::WorkspaceWrite,
                        auto_approve: true,
                        attachments: vec![],
                        worktree: None,
                        resume: None,
                    },
                },
            )
            .unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
        loop {
            let entries = core
                .doc_host
                .open(CHAT)
                .unwrap()
                .doc()
                .read_entries()
                .unwrap();
            let answers: Vec<_> = entries
                .iter()
                .filter(|e| {
                    e.role == MessageRole::Assistant
                        && e.status.is_some_and(|s| s != MessageStatus::Streaming)
                })
                .collect();
            let status = core.sessions.session_status(CHAT).map(|s| s.status);
            let settled = !matches!(status, Some(SessionStatus::Working));
            let timed_out = tokio::time::Instant::now() > deadline;
            if (answers.len() > turn && settled) || timed_out {
                eprintln!(
                    "TURN {turn} after {:?} status={status:?} timed_out={timed_out}",
                    started.elapsed()
                );
                let mut text = String::new();
                if let Some(answer) = answers.get(turn) {
                    for part in &answer.parts {
                        match part {
                            MessagePart::Text { text: t, .. } => text.push_str(t),
                            MessagePart::Error { message, .. } => eprintln!("  ERROR {message}"),
                            _ => {}
                        }
                    }
                }
                eprintln!("  text {text:?}");
                if timed_out || !text.contains("PONG") {
                    failures.push(turn);
                }
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    let (events, _) = core.sessions.subscribe(CHAT, 0).unwrap();
    for event in &events {
        eprintln!("EV {event:?}");
    }
    core.shutdown().await;
    assert!(failures.is_empty(), "turns without PONG: {failures:?}");
}
