// Copyright 2026 Aizen authors. Licensed under Apache-2.0.
//! Tests for auto skill learning in [`super`]: the pure refine/save/skip decision, the real
//! `maybe_run_secretary` flow driven by a replay tape, and the ignored live S4 measurements.

use super::*;

#[test]
fn fuzzy_target_refines_only_what_the_judge_named() {
    use SkillAction::*;
    assert_eq!(skill_action(None, None, false), Save);
    assert_eq!(skill_action(None, Some("b"), false), Refine("b"));
    assert_eq!(skill_action(Some("a"), None, true), Refine("a"));
    assert_eq!(skill_action(Some("a"), Some("b"), false), Skip);
    assert_eq!(skill_action(None, None, true), Save);
}

// --- Design C offline tests (INBOX 13) ---

/// R15: eval scorer and production must agree on the same offer set.
#[test]
fn offer_candidates_agrees_with_eval_shortlist_plus_floor() {
    use crate::persona::secretary;
    let long_steps = "step. ".repeat(500); // ~3000 chars
    let skills = vec![
        secretary::SkillView {
            key: "short",
            when: "when deploying",
            steps: "run deploy",
        },
        secretary::SkillView {
            key: "long",
            when: "when deploying",
            steps: &long_steps,
        },
        secretary::SkillView {
            key: "unrelated",
            when: "when cooking",
            steps: "cook food",
        },
    ];
    let proposal_steps = "deploy app"; // short → floor blocks "long"
    let offered = secretary::offer_candidates(&skills, "when deploying", proposal_steps);
    // Should include "short" (passes floor) but not "long" (floor blocks) or "unrelated" (no match)
    assert_eq!(offered.len(), 1);
    assert_eq!(offered[0].key, "short");
}

/// R22: final_verdict with non-empty missing must return None (Save), not Some(key).
/// Proves eval and production agree: both use final_verdict which checks missing.
#[test]
fn final_verdict_nonempty_missing_returns_none() {
    use crate::persona::secretary;
    let offered = [secretary::SkillView {
        key: "deploy",
        when: "w",
        steps: "s",
    }];
    // Non-empty missing → Save (None)
    assert_eq!(
        secretary::final_verdict(r#"{"same":"deploy","missing":["backup"]}"#, &offered),
        None,
        "non-empty missing must prevent refine"
    );
    // Empty missing → Refine (Some)
    assert_eq!(
        secretary::final_verdict(r#"{"same":"deploy","missing":[]}"#, &offered),
        Some("deploy"),
        "empty missing allows refine"
    );
    // Missing absent → Save (None)
    assert_eq!(
        secretary::final_verdict(r#"{"same":"deploy"}"#, &offered),
        None,
        "absent missing field prevents refine"
    );
}

/// Every way a judge reply can come back, and whether it may refine. The length floor is the
/// other gate; it runs before the call (`offer_candidates_agrees_with_eval_shortlist_plus_floor`).
#[test]
fn final_verdict_refines_only_on_a_named_candidate_with_nothing_missing() {
    use crate::persona::secretary::{final_verdict, SkillView};
    let offered = [SkillView {
        key: "b",
        when: "w",
        steps: "s",
    }];
    for (reply, want) in [
        (r#"{"same":"b","missing":[]}"#, Some("b")),
        (r#"{"same":"b","missing":["x"]}"#, None),
        (r#"{"same":"b"}"#, None),
        (r#"{"same":"b","missing":null}"#, None),
        (r#"{"same":"b","missing":["ok",2]}"#, None),
        (r#"{"same":"b","missing":["","  "]}"#, Some("b")),
        (r#"{"same":"invented","missing":[]}"#, None),
        (r#"{"same":null,"missing":[]}"#, None),
        ("not json", None),
    ] {
        assert_eq!(final_verdict(reply, &offered), want, "{reply}");
    }
}

#[test]
fn parse_same_skill_with_missing_valid_array() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "when deploying",
        steps: "run deploy",
    }];
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"deploy","missing":[]}"#,
        &c,
    );
    assert_eq!(key, Some("deploy"));
    assert_eq!(missing, Some(vec![]));
}

#[test]
fn parse_same_skill_with_missing_nonempty_list() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "when deploying",
        steps: "run deploy",
    }];
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"deploy","missing":["verify checksum","notify team"]}"#,
        &c,
    );
    assert_eq!(key, Some("deploy"));
    assert_eq!(
        missing,
        Some(vec!["verify checksum".into(), "notify team".into()])
    );
}

#[test]
fn parse_same_skill_with_missing_blank_entries_skipped() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "w",
        steps: "s",
    }];
    let (_, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"deploy","missing":["step","  ",""]}"#,
        &c,
    );
    assert_eq!(missing, Some(vec!["step".into()]));
}

#[test]
fn parse_same_skill_with_missing_absent_field_returns_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "w",
        steps: "s",
    }];
    let (key, missing) =
        crate::persona::secretary::parse_same_skill_with_missing(r#"{"same":"deploy"}"#, &c);
    assert_eq!(key, Some("deploy"));
    assert_eq!(missing, None);
}

#[test]
fn parse_same_skill_with_missing_wrong_type_returns_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "w",
        steps: "s",
    }];
    let (_, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"deploy","missing":"not an array"}"#,
        &c,
    );
    assert_eq!(missing, None);
}

#[test]
fn parse_same_skill_with_missing_non_string_element_returns_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "deploy",
        when: "w",
        steps: "s",
    }];
    let (_, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"deploy","missing":["ok",42]}"#,
        &c,
    );
    assert_eq!(missing, None);
}

// --- R18: parser edge cases ---

#[test]
fn parse_missing_fenced_json() {
    let c = [crate::persona::secretary::SkillView {
        key: "k",
        when: "w",
        steps: "s",
    }];
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        "```json\n{\"same\":\"k\",\"missing\":[\"step1\"]}\n```",
        &c,
    );
    assert_eq!(key, Some("k"));
    assert_eq!(missing, Some(vec!["step1".into()]));
}

#[test]
fn parse_missing_prose_around_json() {
    let c = [crate::persona::secretary::SkillView {
        key: "k",
        when: "w",
        steps: "s",
    }];
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        "The skills match. {\"same\":\"k\",\"missing\":[]}",
        &c,
    );
    assert_eq!(key, Some("k"));
    assert_eq!(missing, Some(vec![]));
}

#[test]
fn parse_missing_null_field_returns_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "k",
        when: "w",
        steps: "s",
    }];
    let (_, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"k","missing":null}"#,
        &c,
    );
    assert_eq!(missing, None);
}

#[test]
fn parse_missing_int_field_returns_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "k",
        when: "w",
        steps: "s",
    }];
    let (_, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"k","missing":42}"#,
        &c,
    );
    assert_eq!(missing, None);
}

#[test]
fn parse_invented_key_returns_none_for_key() {
    let c = [crate::persona::secretary::SkillView {
        key: "real",
        when: "w",
        steps: "s",
    }];
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(
        r#"{"same":"invented","missing":[]}"#,
        &c,
    );
    assert_eq!(key, None);
    assert_eq!(missing, Some(vec![])); // missing is valid even if key is invented
}

#[test]
fn parse_over_64kib_returns_none_none() {
    let c = [crate::persona::secretary::SkillView {
        key: "k",
        when: "w",
        steps: "s",
    }];
    let huge = format!(
        "{{\"same\":\"k\",\"missing\":[\"{}\"]}}",
        "x".repeat(70_000)
    );
    let (key, missing) = crate::persona::secretary::parse_same_skill_with_missing(&huge, &c);
    assert_eq!(key, None);
    assert_eq!(missing, None);
}

// --- R19: prompt guard ---

#[test]
fn same_skill_prompt_requires_missing_field_and_format() {
    let prompt = crate::persona::secretary::same_skill_prompt();
    assert!(prompt.contains("MISSING"), "must emphasize MISSING in caps");
    assert!(
        prompt.contains(r#""missing""#),
        "must show the exact JSON field name"
    );
    assert!(
        prompt.contains(r#""same""#),
        "must show the same field name too"
    );
    assert!(
        prompt.contains("equivalent replacement") || prompt.contains("equivalently replaced"),
        "must clarify that equivalent replacements are NOT missing"
    );
}

// --- R20: collapsed_char_len trailing whitespace ---

#[test]
fn collapsed_char_len_trims_trailing_whitespace() {
    use crate::persona::secretary::refine_floor_passes;
    // "ab   " should collapse to "ab" (2 chars). 2/20 = 0.10 → fails.
    // If trailing ws were kept as one space: "ab " = 3 chars. 3/20 = 0.15 → would pass.
    assert!(
        !refine_floor_passes("ab   ", &"x".repeat(20)),
        "trailing whitespace must be trimmed, not collapsed to one space"
    );
    // A run of inner whitespace counts as exactly one space.
    assert_eq!(
        crate::persona::secretary::collapsed_char_len(
            " a  b
	 c "
        ),
        5
    );
}

#[test]
fn refine_floor_passes_ratio_boundary() {
    use crate::persona::secretary::refine_floor_passes;
    // 15% exactly → passes
    assert!(refine_floor_passes(&"x".repeat(15), &"x".repeat(100)));
    // Just below 15% → fails
    assert!(!refine_floor_passes(&"x".repeat(14), &"x".repeat(100)));
    // Empty existing → always passes
    assert!(refine_floor_passes("anything", ""));
    // Equal length → passes
    assert!(refine_floor_passes("abc", "def"));
}

#[test]
fn refine_floor_collapses_whitespace() {
    use crate::persona::secretary::refine_floor_passes;
    // "a  b  c" collapses to "a b c" (5 chars); "abcdefghij" is 10 chars → 50% → passes
    assert!(refine_floor_passes("a  b  c", "abcdefghij"));
    // Leading/trailing whitespace: "  x  " collapses to "x " (2 chars, trailing ws kept once)
    // vs "xxxxxx" (6 chars) → 2/6 ≈ 33% → passes
    assert!(refine_floor_passes("  x  ", "xxxxxx"));
    // "  x  " (2 chars) vs "xxxxxxxxxxxxxx" (14 chars) → 2/14 ≈ 14.3% → fails
    assert!(!refine_floor_passes("  x  ", "xxxxxxxxxxxxxx"));
}

#[test]
fn refine_floor_uses_chars_not_bytes_for_vietnamese() {
    use crate::persona::secretary::refine_floor_passes;
    // R17: ASCII proposal (1 char = 1 byte) vs multi-byte existing.
    // Existing: 100 Vietnamese chars "à" (2 bytes each) = 100 chars, 200 bytes.
    // Proposal: 15 ASCII chars = 15 chars, 15 bytes.
    // Char ratio = 15/100 = 0.15 → passes.
    // Byte ratio = 15/200 = 0.075 → would FAIL if using bytes.
    let existing_vi = &"à".repeat(100); // 100 chars, 200 bytes
    let proposal_ascii = "abcdefghijklmno"; // 15 chars, 15 bytes
    assert!(
        refine_floor_passes(proposal_ascii, existing_vi),
        "char ratio 0.15 must pass even though byte ratio 0.075 would fail"
    );
    // Confirm it's not vacuously passing: 14 ASCII chars vs 100 VI chars → 0.14 < 0.15 → fails
    let proposal_shorter = "abcdefghijklmn"; // 14 chars
    assert!(!refine_floor_passes(proposal_shorter, existing_vi));
}

#[test]
fn same_skill_prompt_mentions_missing_field() {
    let prompt = crate::persona::secretary::same_skill_prompt();
    assert!(
        prompt.contains("missing"),
        "prompt must mention the missing field"
    );
    assert!(
        prompt.contains("MISSING"),
        "prompt must emphasize MISSING in caps"
    );
}

/// The real `maybe_run_secretary` flow with model replies from a replay tape (no network):
/// tape line 0 answers the secretary, line 1 (when present) answers the same-skill judge. An
/// exhausted tape makes the judge call fail, which is the error/timeout path.
mod replay_flow {
    use super::*;
    use crate::core::config::EnvGuard;

    const EN_WHEN: &str = "Khi người dùng muốn dịch tài liệu sang tiếng Anh";
    const EN_STEPS: &str = "Đọc file gốc, dịch từng đoạn sang tiếng Anh, giữ nguyên định dạng markdown, ghi ra file mới cạnh file gốc";
    const VI_WHEN: &str = "Khi người dùng muốn dịch tài liệu sang tiếng Việt";
    const VI_STEPS: &str = "Đọc file gốc, dịch từng đoạn sang tiếng Việt, giữ nguyên định dạng markdown, ghi ra file mới cạnh file gốc";

    fn tape_line(ordinal: usize, content: &str) -> String {
        serde_json::json!({"ordinal": ordinal, "turn": {"content": content}}).to_string()
    }

    /// Seed `dich-tieng-anh`, replay one turn whose secretary proposes `name` (action `new`),
    /// and return every stored skill as (key, body) afterwards.
    async fn run(name: &str, judge: Option<&str>) -> Vec<(String, String)> {
        run_with(
            &[("dich-tieng-anh", EN_WHEN, EN_STEPS)],
            name,
            VI_WHEN,
            VI_STEPS,
            judge,
        )
        .await
        .0
    }

    /// Seed `skills`, replay a secretary proposal, then optionally a judge reply. Returns the
    /// stored skills and how many judge replies the run CONSUMED from the tape (0 = never called).
    #[expect(
        clippy::await_holding_lock,
        reason = "isolated skill home and replay tape across the secretary await"
    )]
    async fn run_with(
        skills: &[(&str, &str, &str)],
        name: &str,
        when: &str,
        steps: &str,
        judge: Option<&str>,
    ) -> (Vec<(String, String)>, usize) {
        let _home = crate::core::config::TEST_HOME_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _tape = crate::llm::replay::TAPE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir();
        // The skill store refuses symlinked paths, and macOS temp lives under the /var symlink.
        // Not on Windows: its canonical form is a verbatim `\\?\` path the store also refuses.
        #[cfg(unix)]
        let tmp = std::fs::canonicalize(tmp).unwrap();
        let root = tmp.join(format!(
            "aizen-skill-flow-{}-{}",
            std::process::id(),
            crate::core::persist::unique_sequence()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let _env = EnvGuard::set([
            ("AIZEN_HOME", root.as_os_str()),
            ("AIZEN_PROJECT_ROOT", root.as_os_str()),
        ]);
        for (key, w, st) in skills {
            skill::save(key, "", w, st).unwrap();
        }
        let secretary = serde_json::json!({"skill": {
            "name": name, "when": when, "steps": steps, "action": "new"
        }})
        .to_string();
        let mut lines = vec![tape_line(0, &secretary)];
        if let Some(j) = judge {
            lines.push(tape_line(1, j));
        }
        let tape = root.join("tape.jsonl");
        std::fs::write(
            &tape,
            lines.join(
                "
",
            ),
        )
        .unwrap();
        crate::llm::replay::configure(crate::llm::replay::Mode::Replay, &tape, None).unwrap();
        let history = vec![
            Message::user("remember that this is how I translate docs".to_string()),
            Message::assistant("Done.".to_string()),
        ];
        let http = http_client().unwrap();
        maybe_run_secretary(&history, &http, "http://127.0.0.1:9", "k", "m").await;
        let consumed = crate::llm::replay::status().map_or(0, |st| st.position);
        crate::llm::replay::disable();
        let mut out: Vec<_> = skill::list()
            .iter()
            .map(|s| (s.key().to_string(), s.body.trim().to_string()))
            .collect();
        out.sort();
        let _ = std::fs::remove_dir_all(&root);
        (out, consumed.saturating_sub(1))
    }

    fn keys(v: &[(String, String)]) -> Vec<&str> {
        v.iter().map(|(k, _)| k.as_str()).collect()
    }

    #[tokio::test]
    async fn judge_null_saves_a_new_skill() {
        let s = run("dich-tieng-viet", Some(r#"{"same":null}"#)).await;
        assert_eq!(keys(&s), ["dich-tieng-anh", "dich-tieng-viet"]);
        assert_eq!(s[0].1, EN_STEPS, "the English skill is untouched");
    }

    #[tokio::test]
    async fn judge_naming_the_candidate_refines_exactly_it() {
        // Updated to new reply format (was `{"same":"dich-tieng-anh"}`); missing=[] required for refine.
        let s = run(
            "dich-tieng-viet",
            Some(r#"{"same":"dich-tieng-anh","missing":[]}"#),
        )
        .await;
        assert_eq!(keys(&s), ["dich-tieng-anh"]);
        assert_eq!(s[0].1, VI_STEPS, "the named skill now carries the proposal");
    }

    #[tokio::test]
    async fn judge_inventing_a_key_saves_a_new_skill() {
        let s = run("dich-tieng-viet", Some(r#"{"same":"invented-key"}"#)).await;
        assert_eq!(keys(&s), ["dich-tieng-anh", "dich-tieng-viet"]);
        assert_eq!(s[0].1, EN_STEPS);
    }

    #[tokio::test]
    async fn judge_call_failure_saves_a_new_skill() {
        let s = run("dich-tieng-viet", None).await;
        assert_eq!(keys(&s), ["dich-tieng-anh", "dich-tieng-viet"]);
        assert_eq!(s[0].1, EN_STEPS);
    }

    #[tokio::test]
    async fn exact_name_without_refine_request_is_skipped() {
        let s = run("dich-tieng-anh", None).await;
        assert_eq!(keys(&s), ["dich-tieng-anh"]);
        assert_eq!(
            s[0].1, EN_STEPS,
            "an exact key is never overwritten unasked"
        );
    }
    fn long_body(min_units: usize) -> String {
        let mut s = String::from(EN_STEPS);
        while crate::persona::secretary::judge_field_units(&s) <= min_units {
            s.push_str(" Dịch từng đoạn sang tiếng Anh, giữ nguyên định dạng markdown.");
        }
        s
    }

    #[tokio::test]
    async fn proposal_over_budget_is_saved_without_a_judge_call() {
        let huge = long_body(crate::persona::secretary::JUDGE_FIELD_UNITS);
        let (s, calls) = run_with(
            &[("dich-tieng-anh", EN_WHEN, EN_STEPS)],
            "dich-tieng-anh-dai",
            EN_WHEN,
            &huge,
            Some(r#"{"same":"dich-tieng-anh"}"#),
        )
        .await;
        assert_eq!(calls, 0, "the judge reply on the tape must stay unread");
        assert_eq!(keys(&s), ["dich-tieng-anh", "dich-tieng-anh-dai"]);
        assert_eq!(s[0].1, EN_STEPS);
    }

    #[tokio::test]
    async fn candidate_over_budget_is_not_refined_even_if_named() {
        let huge = long_body(crate::persona::secretary::JUDGE_FIELD_UNITS);
        let (s, calls) = run_with(
            &[
                ("dich-dai", EN_WHEN, &huge),
                ("dich-ngan", EN_WHEN, EN_STEPS),
            ],
            "translate-doc-english",
            EN_WHEN,
            EN_STEPS,
            Some(r#"{"same":"dich-dai"}"#),
        )
        .await;
        assert_eq!(calls, 1, "the short candidate is still judged");
        assert_eq!(
            keys(&s),
            ["dich-dai", "dich-ngan", "translate-doc-english"],
            "a reply naming the unoffered long skill saves a new skill"
        );
        assert_eq!(s[0].1, huge.trim(), "the long skill is untouched");
    }

    #[tokio::test]
    async fn short_candidate_beside_a_long_one_is_refined_when_named() {
        let huge = long_body(crate::persona::secretary::JUDGE_FIELD_UNITS);
        let (s, calls) = run_with(
            &[
                ("dich-dai", EN_WHEN, &huge),
                ("dich-ngan", EN_WHEN, EN_STEPS),
            ],
            "translate-doc-english",
            EN_WHEN,
            VI_STEPS,
            Some(r#"{"same":"dich-ngan","missing":[]}"#),
        )
        .await;
        assert_eq!(calls, 1);
        assert_eq!(keys(&s), ["dich-dai", "dich-ngan"]);
        assert_eq!(s[1].1, VI_STEPS, "the short candidate carries the proposal");
    }

    /// Design C R16: candidates that PASS the floor but exceed the token budget are cut.
    /// The judge is called with only the ones that fit; naming a cut candidate saves new.
    #[tokio::test]
    async fn candidate_cut_by_token_budget_is_not_refined_even_if_named() {
        // Each candidate ~95k units. Two fill the 200k budget; third is cut by budget.
        // Proposal is similar length → passes floor for all three.
        let big = long_body(95_000);
        let proposal_big = long_body(90_000); // similar length → floor passes
        let (s, calls) = run_with(
            &[
                ("dich-a", EN_WHEN, &big),
                ("dich-b", EN_WHEN, &big),
                ("dich-c", EN_WHEN, &big),
            ],
            "translate-doc-english",
            EN_WHEN,
            &proposal_big,
            Some(r#"{"same":"dich-c","missing":[]}"#),
        )
        .await;
        assert_eq!(calls, 1, "judge called with budget-fitting candidates");
        assert_eq!(
            keys(&s),
            ["dich-a", "dich-b", "dich-c", "translate-doc-english"],
            "naming the cut candidate saves new"
        );
    }

    /// Design C R16b: floor blocks oversized candidates before the judge is called.
    #[tokio::test]
    async fn floor_blocks_tiny_proposal_against_long_existing_replay() {
        let big = long_body(95_000);
        let (s, calls) = run_with(
            &[("dich-a", EN_WHEN, &big), ("dich-b", EN_WHEN, &big)],
            "tiny-skill",
            EN_WHEN,
            "do x", // tiny vs 95k → floor blocks
            Some(r#"{"same":"dich-a","missing":[]}"#),
        )
        .await;
        assert_eq!(
            calls, 0,
            "floor blocks all oversized candidates; no judge call"
        );
        assert_eq!(keys(&s), ["dich-a", "dich-b", "tiny-skill"]);
    }

    /// Design C: judge says same but lists a missing step → saved new, old body untouched.
    #[tokio::test]
    async fn judge_missing_step_saves_new_and_preserves_old_body() {
        let (s, calls) = run_with(
            &[("dich-tieng-anh", EN_WHEN, EN_STEPS)],
            "translate-doc-english",
            EN_WHEN,
            EN_STEPS,
            Some(r#"{"same":"dich-tieng-anh","missing":["update CHANGELOG"]}"#),
        )
        .await;
        assert_eq!(calls, 1);
        assert_eq!(keys(&s), ["dich-tieng-anh", "translate-doc-english"]);
        assert_eq!(
            s[0].1, EN_STEPS,
            "old skill body must be byte-identical when missing steps reported"
        );
    }

    /// Design C: old-format reply (no missing field) → missing=None → saved new.
    #[tokio::test]
    async fn old_format_reply_without_missing_saves_new() {
        let (s, calls) = run_with(
            &[("dich-tieng-anh", EN_WHEN, EN_STEPS)],
            "translate-doc-english",
            EN_WHEN,
            EN_STEPS,
            Some(r#"{"same":"dich-tieng-anh"}"#),
        )
        .await;
        assert_eq!(calls, 1);
        assert_eq!(keys(&s), ["dich-tieng-anh", "translate-doc-english"]);
        assert_eq!(
            s[0].1, EN_STEPS,
            "old skill body must be byte-identical for old-format reply"
        );
    }

    /// Design C: tiny proposal vs long existing → floor blocks, no judge call, saved new.
    #[tokio::test]
    async fn floor_blocks_tiny_proposal_against_long_existing() {
        let long_steps = "step one. ".repeat(200); // ~1100 chars
        let (s, calls) = run_with(
            &[("big-skill", EN_WHEN, &long_steps)],
            "tiny-skill",
            EN_WHEN,
            "do x", // 4 chars vs ~1100 → ratio ≈ 0.004 < 0.15
            Some(r#"{"same":"big-skill","missing":[]}"#),
        )
        .await;
        assert_eq!(calls, 0, "floor must block before the judge is called");
        assert_eq!(keys(&s), ["big-skill", "tiny-skill"]);
    }
}

/// Live S4 measurement against the configured chore model, through the same shortlist → judge →
/// parse path production uses. Ignored by default (network + cost):
/// `cargo test --bin aizen s4_live_judge_eval -- --ignored --nocapture`
/// (`AIZEN_EVAL_RUNS`, default 3; `AIZEN_EVAL_EXPECT_MODEL` fails the run on any other model).
mod s4_live_eval {
    use super::*;
    use crate::persona::secretary;
    use crate::persona::skill_overlap_eval::{
        blind_cases, cases, coverage_cases, holdout_cases, length_tier_cases, multi_cases, score,
        Case, MAX_EXTRA_SKILL, MAX_EXTRA_SKILL_BLIND,
    };

    async fn judge_sets(sets: Vec<(&str, Vec<Case>)>) {
        let (base, key, model) = resolve_endpoint(None, None, None).expect("endpoint");
        let ep = summarizer_endpoint(&base, &key, &model);
        println!("judge model: {} @ {}", ep.model, ep.base_url);
        if let Ok(want) = std::env::var("AIZEN_EVAL_EXPECT_MODEL") {
            assert_eq!(ep.model, want, "judge routed to an unexpected model");
        }
        let http = http_client().expect("http");
        let runs: usize = std::env::var("AIZEN_EVAL_RUNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let mut violations: Vec<String> = Vec::new();
        for (set, cs) in sets {
            for run in 1..=runs {
                let mut verdicts: Vec<Option<String>> = Vec::new();
                let (mut calls, mut ms, mut prompt, mut completion) = (0u64, 0u128, 0u64, 0u64);
                let (mut total, mut mismatch, mut missing) = (0u64, 0u64, 0u64);
                for c in &cs {
                    let offered = secretary::offer_candidates(
                        &c.store(),
                        &c.proposal.when,
                        &c.proposal.steps,
                    );
                    if offered.is_empty() {
                        println!(
                            "  usage {:32} offered=0 missing=[] final=Save (floor_blocked)",
                            c.id
                        );
                        verdicts.push(None);
                        continue;
                    }
                    let messages =
                        same_skill_messages(&c.proposal.when, &c.proposal.steps, &offered);
                    let t = std::time::Instant::now();
                    let reply =
                        chore_chat(&http, &ep.base_url, &ep.api_key, &ep.model, &messages, &[])
                            .await;
                    ms += t.elapsed().as_millis();
                    calls += 1;
                    let verdict = match reply {
                        Ok(r) => {
                            // Provider-reported usage, never an estimate; "-" when not sent.
                            let u = r.usage.as_ref();
                            let p = u.and_then(|u| u.prompt_tokens);
                            let k = u.and_then(|u| u.completion_tokens);
                            let tot = u.and_then(|u| u.total_tokens);
                            // A missing field is counted, never summed as 0, so totals stay honest.
                            let bad_total =
                                matches!((p, k, tot), (Some(p), Some(k), Some(t)) if t != p + k);
                            match (p, k, tot) {
                                (Some(p), Some(k), Some(tot)) => {
                                    prompt += p;
                                    completion += k;
                                    total += tot;
                                    mismatch += u64::from(bad_total);
                                }
                                _ => missing += 1,
                            }
                            let f = |v: Option<u64>| v.map_or("-".into(), |n| n.to_string());
                            // Unknown usage keys verbatim (e.g. reasoning-token splits).
                            let extra = u
                                .map(|u| serde_json::to_string(&u.extra).unwrap_or_default())
                                .unwrap_or_else(|| "no usage".into());
                            let raw_content = r.content.unwrap_or_default();
                            let (_, parsed_missing) =
                                secretary::parse_same_skill_with_missing(&raw_content, &offered);
                            let verdict = secretary::final_verdict(&raw_content, &offered);
                            let final_str =
                                verdict.map_or("Save".into(), |k| format!("Refine({k})"));
                            let missing_str = match &parsed_missing {
                                Some(m) if m.is_empty() => "[]".into(),
                                Some(m) => format!("{:?}", m),
                                None => "null".into(),
                            };
                            println!(
                                "  usage {:32} offered={} missing={missing_str} final={final_str} prompt={} completion={} total={}{} extra={extra}",
                                c.id,
                                offered.len(),
                                f(p),
                                f(k),
                                f(tot),
                                if bad_total { " TOTAL!=P+C" } else { "" },
                            );
                            verdict.map(str::to_string)
                        }
                        Err(e) => {
                            println!("  call failed {}: {e}", c.id);
                            None
                        }
                    };
                    verdicts.push(verdict);
                }
                println!(
                    "--- {set} run {run}: calls={calls}/{} prompt_tokens={prompt} completion_tokens={completion} total_tokens={total} total_ne_p_plus_c={mismatch} usage_missing={missing} avg_ms={}",
                    cs.len(),
                    ms / u128::from(calls.max(1))
                );
                let by_id: std::collections::HashMap<&str, Option<String>> =
                    cs.iter().map(|c| c.id.as_str()).zip(verdicts).collect();
                let s = score(&format!("s4 {set} r{run}"), &cs, |c| {
                    by_id[c.id.as_str()].clone()
                });
                let max_extra = if set == "blind" {
                    MAX_EXTRA_SKILL_BLIND.unwrap_or_else(|| {
                        panic!("MAX_EXTRA_SKILL_BLIND must be set before running a non-empty blind set")
                    })
                } else {
                    MAX_EXTRA_SKILL
                        .iter()
                        .find(|(name, _)| *name == set)
                        .map(|(_, n)| *n)
                        .expect("every set has a fixed threshold")
                };
                if s.wrong_overwrite != 0 || s.extra_skill > max_extra {
                    violations.push(format!(
                        "{set} r{run}: wrong_overwrite={} (max 0), extra_skill={} (max {max_extra})",
                        s.wrong_overwrite, s.extra_skill
                    ));
                }
            }
        }
        // Asserted after every set ran, so a failure still prints all numbers.
        assert!(
            violations.is_empty(),
            "thresholds violated: {violations:#?}"
        );
    }

    #[tokio::test]
    #[ignore]
    async fn s4_live_judge_eval() {
        let mut sets = vec![
            ("single", cases()),
            ("multi", multi_cases()),
            ("coverage", coverage_cases()),
            ("holdout", holdout_cases()),
        ];
        let blind = blind_cases();
        if !blind.is_empty() {
            assert!(
                MAX_EXTRA_SKILL_BLIND.is_some(),
                "blind set has {} cases but MAX_EXTRA_SKILL_BLIND is None — \
                 set the threshold BEFORE running live; never after seeing results",
                blind.len()
            );
            sets.push(("blind", blind));
        }
        judge_sets(sets).await;
    }

    /// Short / medium / long skill bodies (≈0.2k / 0.7k / 3k chars; real store skills are 3.1k and
    /// 9.8k). `cargo test --bin aizen s4_live_length_tiers -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn s4_live_length_tiers() {
        judge_sets(length_tier_cases()).await;
    }

    /// Short proposals written BEFORE the live run (what a secretary typically proposes), one per
    /// real skill: (real key, when, steps).
    const SHORT_PROPOSALS: [(&str, &str, &str); 2] = [
        (
            "nghiep-vu-backend-cua-project",
            "Khi cần sửa route, API hoặc bảng dữ liệu phía backend của project",
            "1. Dùng bộ tool mcp_backend_* tìm đúng route, job hoặc bảng liên quan\n2. Sửa tại chỗ\n3. Gọi lại endpoint để kiểm chứng trước khi báo xong",
        ),
        (
            "thiet-ke-tren-ban-thiet-ke",
            "Khi cần sửa bố cục, màu hoặc chữ trên tài liệu thiết kế của Aizen Desktop",
            "1. Mở tài liệu thiết kế bằng bộ tool mcp_design_*\n2. Sửa đúng phần tử cần đổi, không viết lại cả tệp\n3. Xem lại trang sau khi sửa",
        ),
    ];

    fn skill_file_hashes() -> Vec<(String, String)> {
        let mut out: Vec<_> = std::fs::read_dir(skill::skills_dir())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .map(|e| {
                let bytes = std::fs::read(e.path()).unwrap();
                let d = ring::digest::digest(&ring::digest::SHA256, &bytes);
                let hex: String = d.as_ref().iter().map(|b| format!("{b:02x}")).collect();
                (e.file_name().to_string_lossy().into_owned(), hex)
            })
            .collect();
        out.sort();
        out
    }

    /// Cost on the REAL store, read-only: judge calls only, never refine/save; the skill files'
    /// SHA-256 must be identical before and after. Two proposal shapes per real skill: a full copy
    /// under another name (upper bound: the skill is in the input twice) and a short proposal
    /// (typical). `cargo test --bin aizen s4_live_real_store -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn s4_live_real_store() {
        let before = skill_file_hashes();
        let (base, key, model) = resolve_endpoint(None, None, None).expect("endpoint");
        let ep = summarizer_endpoint(&base, &key, &model);
        println!("judge model: {} @ {}", ep.model, ep.base_url);
        if let Ok(want) = std::env::var("AIZEN_EVAL_EXPECT_MODEL") {
            assert_eq!(ep.model, want, "judge routed to an unexpected model");
        }
        let http = http_client().expect("http");
        let all = skill::list();
        let views = skill_views(&all);
        let runs: usize = std::env::var("AIZEN_EVAL_RUNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        for (real, short_when, short_steps) in SHORT_PROPOSALS {
            let sk = all
                .iter()
                .find(|s| s.key() == real)
                .expect("real skill present");
            for (shape, when, steps) in [
                ("copy (upper bound)", sk.when.as_str(), sk.body.as_str()),
                ("short (typical)", short_when, short_steps),
            ] {
                for run in 1..=runs {
                    let offered = crate::persona::secretary::offer_candidates(&views, when, steps);
                    if offered.is_empty() {
                        // Floor or budget blocked all candidates.
                        let existing = views.iter().find(|v| v.key == real);
                        let ratio = existing.map_or(f64::NAN, |e| {
                            let el = crate::persona::secretary::collapsed_char_len(e.steps);
                            if el == 0 {
                                f64::INFINITY
                            } else {
                                crate::persona::secretary::collapsed_char_len(steps) as f64
                                    / el as f64
                            }
                        });
                        println!(
                            "  {real} | {shape} | r{run}: floor_blocked=true missing=[] final=Save ratio={ratio:.4}",
                        );
                        continue;
                    }
                    let messages = same_skill_messages(when, steps, &offered);
                    let t = std::time::Instant::now();
                    let reply =
                        chore_chat(&http, &ep.base_url, &ep.api_key, &ep.model, &messages, &[])
                            .await;
                    let ms = t.elapsed().as_millis();
                    match reply {
                        Ok(r) => {
                            let u = r.usage.as_ref();
                            let f = |v: Option<u64>| v.map_or("-".into(), |n| n.to_string());
                            let raw = r.content.unwrap_or_default();
                            let (_, missing) =
                                crate::persona::secretary::parse_same_skill_with_missing(
                                    &raw, &offered,
                                );
                            let verdict = crate::persona::secretary::final_verdict(&raw, &offered);
                            let final_action =
                                verdict.map_or("Save".into(), |k| format!("Refine({k})"));
                            let missing_str = match &missing {
                                Some(m) if m.is_empty() => "[]".into(),
                                Some(m) => format!("{:?}", m),
                                None => "null".into(),
                            };
                            let existing = views.iter().find(|v| v.key == real);
                            let ratio = existing.map_or(f64::NAN, |e| {
                                let el = crate::persona::secretary::collapsed_char_len(e.steps);
                                if el == 0 {
                                    f64::INFINITY
                                } else {
                                    crate::persona::secretary::collapsed_char_len(steps) as f64
                                        / el as f64
                                }
                            });
                            println!(
                                "  {real} | {shape} | r{run}: offered={:?} missing={missing_str} final={final_action} ratio={ratio:.4} prompt={} completion={} total={} ms={ms}",
                                offered.iter().map(|s| s.key).collect::<Vec<_>>(),
                                f(u.and_then(|u| u.prompt_tokens)),
                                f(u.and_then(|u| u.completion_tokens)),
                                f(u.and_then(|u| u.total_tokens)),
                            );
                        }
                        Err(e) => {
                            println!("  {real} | {shape} | r{run}: call failed after {ms} ms: {e}")
                        }
                    }
                }
            }
        }
        assert_eq!(
            skill_file_hashes(),
            before,
            "the real store must be untouched"
        );
    }
}
