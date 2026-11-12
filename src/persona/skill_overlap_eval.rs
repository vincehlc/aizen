// Copyright 2026 Aizen authors. Licensed under Apache-2.0.
//! Labeled skill-overlap cases shared by every auto-learn targeting strategy, so strategies are
//! compared on the same data. Each case is one existing skill (plus optional `others` — distractor
//! skills in the same store) and one secretary proposal with a human label: `same` (refining
//! `existing` is right) or `different` (no skill may be overwritten).
//!
//! Metrics: a WRONG OVERWRITE replaces a skill that should have been kept — a `different` case
//! refined at all, or a `same` case refining a distractor (data damage: archived, and announced
//! only as a one-line "refined skill" notice); an EXTRA SKILL keeps a duplicate that should have
//! been refined (store bloat).

use crate::persona::secretary::{
    build_same_skill_input, fits_judge, judge_field_units, offer_candidates, overlap_candidates,
    parse_same_skill_with_missing, SkillView, JUDGE_FIELD_UNITS,
};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub(crate) struct Skill {
    pub key: String,
    pub when: String,
    pub steps: String,
}

#[derive(Deserialize, Clone)]
pub(crate) struct Proposal {
    pub name: String,
    pub when: String,
    pub steps: String,
}

/// A typo in a label fails deserialization instead of silently reading as `different`.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Expect {
    Same,
    Different,
}

#[derive(Deserialize, Clone)]
pub(crate) struct Case {
    pub id: String,
    pub expect: Expect,
    pub existing: Skill,
    #[serde(default)]
    pub others: Vec<Skill>,
    pub proposal: Proposal,
}

fn as_view(s: &Skill) -> SkillView<'_> {
    SkillView {
        key: &s.key,
        when: &s.when,
        steps: &s.steps,
    }
}

impl Case {
    pub fn same(&self) -> bool {
        self.expect == Expect::Same
    }
    /// The store as the gate sees it: distractors FIRST, so a first-match gate is tested against
    /// the case where a wrong skill comes before the right one.
    pub fn store(&self) -> Vec<SkillView<'_>> {
        self.others
            .iter()
            .chain(std::iter::once(&self.existing))
            .map(as_view)
            .collect()
    }
    /// What production offers the judge (lexical shortlist minus anything over the judge budget,
    /// then filtered by the same-floor gate).
    pub fn shortlist(&self) -> Vec<SkillView<'_>> {
        offer_candidates(&self.store(), &self.proposal.when, &self.proposal.steps)
    }
    /// The guard decides this case by code: the proposal or the right skill cannot be judged
    /// whole, so `existing` is never offered and never refined.
    pub fn blocked(&self) -> bool {
        !fits_judge(&self.proposal.when, &self.proposal.steps)
            || !fits_judge(&self.existing.when, &self.existing.steps)
    }
}

/// The 18 single-skill cases.
pub(crate) fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!("testdata/skill_overlap_cases.json")).unwrap()
}

/// Cases whose store holds several similar skills, so the judge must pick the right one.
pub(crate) fn multi_cases() -> Vec<Case> {
    serde_json::from_str(include_str!("testdata/skill_overlap_multi.json")).unwrap()
}

/// Blind cases written by the user/reviewer, never by the implementer. Empty = skip.
pub(crate) fn blind_cases() -> Vec<Case> {
    serde_json::from_str(include_str!("testdata/skill_overlap_blind.json")).unwrap()
}

/// Coverage cases: same procedure, testing whether shorter proposals keep all important steps.
/// 3 shorter-but-complete (expect refine), 3 shorter-and-drops-step (expect NO refine).
pub(crate) fn coverage_cases() -> Vec<Case> {
    serde_json::from_str(include_str!("testdata/skill_overlap_coverage.json")).unwrap()
}

/// Held-out cases written by the reviewer and committed only after the Design C implementation
/// (276860c), so the implementer never saw them while coding.
pub(crate) fn holdout_cases() -> Vec<Case> {
    serde_json::from_str(include_str!("testdata/skill_overlap_holdout.json")).unwrap()
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Score {
    pub correct: usize,
    pub wrong_overwrite: usize,
    pub extra_skill: usize,
}

/// `refined(case)` = the key the strategy overwrites for this proposal, `None` = saved as new.
pub(crate) fn score(
    label: &str,
    cases: &[Case],
    refined: impl Fn(&Case) -> Option<String>,
) -> Score {
    let mut s = Score::default();
    for c in cases {
        match refined(c) {
            Some(k) if c.same() && k == c.existing.key => s.correct += 1,
            Some(k) => {
                s.wrong_overwrite += 1;
                println!("  [{label}] WRONG OVERWRITE  {} (refined {k})", c.id);
            }
            None if c.same() => {
                s.extra_skill += 1;
                println!("  [{label}] EXTRA SKILL      {}", c.id);
            }
            None => s.correct += 1,
        }
    }
    println!(
        "[{label}] n={} correct={} wrong_overwrite={} extra_skill={}",
        cases.len(),
        s.correct,
        s.wrong_overwrite,
        s.extra_skill
    );
    s
}

/// The gate shipped before S4, frozen here so the baseline stays comparable: first skill in store
/// order with BOTH axes over threshold, auto-refined.
pub(crate) fn legacy_target(c: &Case) -> Option<String> {
    use crate::core::text_match::match_similarity as sim;
    c.store()
        .into_iter()
        .find(|s| sim(&c.proposal.when, s.when) >= 0.45 && sim(&c.proposal.steps, s.steps) >= 0.35)
        .map(|s| s.key.to_string())
}

/// S4 with a PERFECT judge: refines `existing` exactly when it is shortlisted and truly the same.
fn ceiling_target(c: &Case) -> Option<String> {
    (c.same() && c.shortlist().iter().any(|s| s.key == c.existing.key))
        .then(|| c.existing.key.clone())
}

#[test]
fn fixtures_are_well_formed() {
    // Exact sizes and label splits: a case lost in a merge fails here, not as a moved score.
    for (cs, same, different) in [(cases(), 8, 10), (multi_cases(), 5, 2)] {
        assert_eq!(cs.iter().filter(|c| c.same()).count(), same);
        assert_eq!(cs.iter().filter(|c| !c.same()).count(), different);
        // No proposal reuses a stored key AS PRODUCTION SEES IT (sanitized): exact-name
        // refinement is a different path and is not what is being measured.
        assert!(cs.iter().all(|c| {
            let key = crate::skills::sanitize_name(&c.proposal.name);
            c.store().iter().all(|s| s.key != key)
        }));
    }
    assert!(multi_cases().iter().all(|c| !c.others.is_empty()));
    // `score` reports by id, so a duplicated id would hide which case failed.
    let mut ids: Vec<String> = cases()
        .into_iter()
        .chain(multi_cases())
        .chain(length_tier_cases().into_iter().flat_map(|(_, cs)| cs))
        .chain(blind_cases())
        .map(|c| c.id)
        .collect();
    let n = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), n, "case ids must be unique");
}

/// Reusable well-formedness checks for any case set: unique ids, no key collision.
fn assert_cases_well_formed(cs: &[Case], label: &str) {
    assert!(
        cs.iter().all(|c| {
            let key = crate::skills::sanitize_name(&c.proposal.name);
            c.store().iter().all(|s| s.key != key)
        }),
        "{label}: proposal name must not collide with any stored key"
    );
    let mut ids: Vec<&str> = cs.iter().map(|c| c.id.as_str()).collect();
    let n = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), n, "{label}: case ids must be unique");
}

/// Offline sanity check for the blind set. Empty = skip (no cases yet).
#[test]
fn blind_cases_offline() {
    let cs = blind_cases();
    if cs.is_empty() {
        println!("  blind_cases: empty — no cases written yet, skipping.");
        return;
    }
    assert_cases_well_formed(&cs, "blind");
    let s = score("blind", &cs, ceiling_target);
    println!(
        "  blind_cases: {} cases, score={:?}, MAX_EXTRA_SKILL_BLIND={:?}",
        cs.len(),
        s,
        MAX_EXTRA_SKILL_BLIND
    );
}

/// Prove duplicate-id detection works without touching the real file.
#[test]
fn duplicate_id_is_rejected() {
    let json = r#"[
        {"id":"dup","expect":"same","existing":{"key":"a","when":"w","steps":"s"},"proposal":{"name":"b","when":"w","steps":"s"}},
        {"id":"dup","expect":"different","existing":{"key":"c","when":"w","steps":"s"},"proposal":{"name":"d","when":"w","steps":"s"}}
    ]"#;
    let cs: Vec<Case> = serde_json::from_str(json).unwrap();
    let result = std::panic::catch_unwind(|| assert_cases_well_formed(&cs, "dup-test"));
    assert!(result.is_err(), "duplicate ids must be rejected");
}

/// Offline well-formedness check for coverage cases. Must have unique ids, no key collision,
/// and at least 6 cases (3 shorter-complete + 3 shorter-drops-step).
#[test]
fn coverage_cases_offline() {
    let cs = coverage_cases();
    assert!(
        cs.len() >= 6,
        "coverage_cases must have at least 6 cases, got {}",
        cs.len()
    );
    assert_cases_well_formed(&cs, "coverage");
    // At least 2 cases with long Vietnamese bodies (> 2000 chars).
    for c in &cs {
        println!(
            "  {} existing.chars={} proposal.chars={}",
            c.id,
            c.existing.steps.chars().count(),
            c.proposal.steps.chars().count()
        );
    }
    let long_vi = cs
        .iter()
        .filter(|c| {
            c.existing.steps.chars().count() > 2000 || c.proposal.steps.chars().count() > 2000
        })
        .count();
    assert!(
        long_vi >= 2,
        "need at least 2 cases with long Vietnamese bodies (>2000 chars), got {}",
        long_vi
    );
    println!(
        "  coverage_cases: {} cases ({} with long Vi bodies)",
        cs.len(),
        long_vi
    );
}

/// Baseline: the pre-S4 behaviour, where every lexical match is auto-refined.
#[test]
fn baseline_lexical_auto_refine() {
    for c in &cases() {
        println!(
            "  {:32} when={:.2} steps={:.2} expect={:?}",
            c.id,
            crate::core::text_match::match_similarity(&c.proposal.when, &c.existing.when),
            crate::core::text_match::match_similarity(&c.proposal.steps, &c.existing.steps),
            c.expect
        );
    }
    // Pinned so a change to text_match that moves these numbers is noticed, not silent.
    assert_eq!(
        score("baseline", &cases(), legacy_target),
        Score {
            correct: 9,
            wrong_overwrite: 6,
            extra_skill: 3
        }
    );
    assert_eq!(
        score("baseline-multi", &multi_cases(), legacy_target),
        Score {
            correct: 3,
            wrong_overwrite: 2,
            extra_skill: 2
        }
    );
}

/// The ceiling the shortlist allows. With either axis enough to shortlist, every labeled duplicate
/// reaches the judge; under the old both-axes gate 3 of 8 never did (ceiling was 15/0/3).
#[test]
fn s4_ceiling_with_a_perfect_judge() {
    assert_eq!(
        score("s4-ceiling", &cases(), ceiling_target),
        Score {
            correct: 18,
            wrong_overwrite: 0,
            extra_skill: 0
        }
    );
    assert_eq!(
        score("s4-ceiling-multi", &multi_cases(), ceiling_target),
        Score {
            correct: 7,
            wrong_overwrite: 0,
            extra_skill: 0
        }
    );
}

fn view<'a>(key: &'a str) -> SkillView<'a> {
    SkillView {
        key,
        when: "verify the build",
        steps: "Run cargo test before release.",
    }
}

#[test]
fn judge_reply_must_name_an_offered_candidate() {
    let c = [view("release-check")];
    assert_eq!(
        parse_same_skill_with_missing(r#"{"same":"release-check"}"#, &c).0,
        Some("release-check")
    );
    assert_eq!(
        parse_same_skill_with_missing("```json\n{\"same\": \"release-check\"}\n```", &c).0,
        Some("release-check")
    );
    assert_eq!(
        parse_same_skill_with_missing(r#"{"same":null}"#, &c).0,
        None
    );
    assert_eq!(
        parse_same_skill_with_missing(r#"{"same":"invented-key"}"#, &c).0,
        None
    );
    assert_eq!(parse_same_skill_with_missing("not json", &c).0, None);
    assert_eq!(parse_same_skill_with_missing("", &c).0, None);
}

#[test]
fn shortlist_is_capped_best_first_and_judge_input_names_every_candidate() {
    let keys = ["a", "b", "c", "d"];
    let mut views: Vec<_> = keys.iter().map(|k| view(k)).collect();
    // `d` is the only exact match, listed last: best-first ordering must still keep it.
    for v in &mut views[..3] {
        v.when = "verify the build quickly";
        v.steps = "Run cargo test.";
    }
    let short = overlap_candidates(&views, "verify the build", "Run cargo test before release.");
    assert_eq!(short.len(), 3);
    assert_eq!(short[0].key, "d");
    let input = build_same_skill_input("verify the build", "Run cargo test.", &short);
    assert!(short
        .iter()
        .all(|s| input.contains(&format!("- key: {}", s.key))));
}

/// One domain of the length-tier generator: generic steps shared by every variant, the ONE step
/// that names the target (placed second to last, where a long body's clip falls), a closing step,
/// a step added by the "extra step" duplicate, and word swaps for the "reworded" duplicate.
struct Domain {
    tag: &'static str,
    when_base: &'static str,
    when_target: fn(&str) -> String,
    pool: &'static [&'static str],
    crucial: fn(&str) -> String,
    last: &'static str,
    extra: &'static str,
    swaps: &'static [(&'static str, &'static str)],
    targets: (&'static str, &'static str),
}

const RELEASE: Domain = Domain {
    tag: "release",
    when_base: "When releasing the backend service",
    when_target: |t| format!("When releasing the backend service to {t}"),
    pool: &[
        "Pull the latest main branch and confirm the working tree is clean",
        "Run cargo fmt --check and fix any formatting drift",
        "Run cargo clippy with -D warnings and resolve every lint",
        "Run the full test suite and keep the log for the release notes",
        "Bump the version in Cargo.toml and regenerate Cargo.lock",
        "Update CHANGELOG.md with every user-facing change since the last tag",
        "Build the release binary and record its size and startup time",
        "Build the docker image and tag it with the new version",
        "Push the image to the container registry",
        "Run the database migrations against a snapshot first",
        "Smoke-test the health endpoint and the login flow",
        "Write the rollback command into the release ticket",
    ],
    crucial: |t| {
        match t {
        "staging" => "Deploy with kubectl apply -f k8s/staging and watch the rollout".into(),
        _ => "Get approval from the on-call lead, then deploy with kubectl apply -f k8s/production and watch the rollout".into(),
    }
    },
    last: "Announce the release in the team channel with the changelog link",
    extra: "Tag the release commit with the new version and push the tag",
    swaps: &[
        ("Run ", "Execute "),
        ("confirm", "verify"),
        ("Push the", "Upload the"),
        ("When releasing", "When shipping"),
    ],
    targets: ("staging", "production"),
};

const TRANSLATE: Domain = Domain {
    tag: "translate",
    when_base: "Khi người dùng muốn dịch và xuất bản tài liệu hướng dẫn",
    when_target: |t| format!("Khi người dùng muốn dịch tài liệu hướng dẫn sang tiếng {t}"),
    pool: &[
        "Đọc toàn bộ file gốc và liệt kê các thuật ngữ chuyên ngành",
        "Kiểm tra bảng thuật ngữ đã có trong thư mục docs/glossary",
        "Giữ nguyên các khối code, đường dẫn file và tên lệnh",
        "Dịch từng đoạn, giữ nguyên định dạng markdown và tiêu đề",
        "Rà lại các câu dài, tách câu nếu cần cho dễ đọc",
        "Kiểm tra lại các liên kết nội bộ vẫn trỏ đúng",
        "Chạy công cụ kiểm tra chính tả cho bản dịch",
        "So sánh số lượng tiêu đề giữa bản gốc và bản dịch",
        "Ghi chú các đoạn chưa chắc nghĩa để hỏi lại người dùng",
        "Cập nhật bảng thuật ngữ với các từ mới",
        "Lưu bản dịch cạnh file gốc với hậu tố ngôn ngữ",
        "Tạo commit riêng cho bản dịch",
    ],
    crucial: |t| match t {
        "Anh" => "Xuất bản bản dịch tiếng Anh lên trang docs/en".into(),
        _ => "Xuất bản bản dịch tiếng Nhật lên trang docs/ja".into(),
    },
    last: "Báo cho người dùng đường dẫn bản dịch đã xuất bản",
    extra: "Gửi bản dịch cho reviewer duyệt trước khi xuất bản",
    swaps: &[
        ("Đọc toàn bộ", "Xem hết"),
        ("Kiểm tra", "Rà soát"),
        ("Khi người dùng muốn", "Khi user cần"),
    ],
    targets: ("Anh", "Nhật"),
};

/// Generic steps per tier: short 1, medium 8, long 38 (cycling the pool as numbered sections).
pub(crate) const TIERS: [(&str, usize); 3] = [("short", 1), ("medium", 8), ("long", 38)];

/// The target-naming step goes after `at` generic steps (`at == generic`: second to last). The
/// extra step, when present, sits right before it.
fn tier_steps(d: &Domain, generic: usize, target: &str, extra: bool, at: usize) -> String {
    let mut steps: Vec<String> = (0..generic)
        .map(|i| {
            let s = d.pool[i % d.pool.len()];
            if generic > d.pool.len() {
                format!("{s} (section {})", i / d.pool.len() + 1)
            } else {
                s.to_string()
            }
        })
        .collect();
    let at = at.min(generic);
    steps.insert(at, (d.crucial)(target));
    if extra {
        steps.insert(at, d.extra.to_string());
    }
    steps.push(d.last.to_string());
    steps
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {s}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn reword(d: &Domain, s: &str) -> String {
    d.swaps
        .iter()
        .fold(s.to_string(), |acc, (a, b)| acc.replace(a, b))
}

/// Per tier and domain: two true duplicates (reworded; one extra step) and two distinct skills
/// (target named in `when`; target named ONLY in the second-to-last step).
pub(crate) fn length_tier_cases() -> Vec<(&'static str, Vec<Case>)> {
    TIERS
        .iter()
        .map(|&(tier, generic)| {
            let mut cs = Vec::new();
            for d in [&RELEASE, &TRANSLATE] {
                let (t1, t2) = d.targets;
                let skill = |when: String, steps: String| Skill {
                    key: format!("{}-{tier}", d.tag),
                    when,
                    steps,
                };
                let mut push =
                    |kind: &str, expect, existing: Skill, when: String, steps: String| {
                        cs.push(Case {
                            id: format!("{tier}-{}-{kind}", d.tag),
                            expect,
                            existing,
                            others: Vec::new(),
                            proposal: Proposal {
                                name: format!("{}-{tier}-{kind}", d.tag),
                                when,
                                steps,
                            },
                        })
                    };
                let base_when = (d.when_target)(t1);
                let late = generic;
                let base_steps = tier_steps(d, generic, t1, false, late);
                push(
                    "same-reworded",
                    Expect::Same,
                    skill(base_when.clone(), base_steps.clone()),
                    reword(d, &base_when),
                    reword(d, &base_steps),
                );
                push(
                    "same-extra-step",
                    Expect::Same,
                    skill(base_when.clone(), base_steps.clone()),
                    base_when.clone(),
                    tier_steps(d, generic, t1, true, late),
                );
                push(
                    "diff-when",
                    Expect::Different,
                    skill(base_when.clone(), base_steps.clone()),
                    (d.when_target)(t2),
                    tier_steps(d, generic, t2, false, late),
                );
                push(
                    "diff-late-step",
                    Expect::Different,
                    skill(d.when_base.to_string(), base_steps.clone()),
                    d.when_base.to_string(),
                    tier_steps(d, generic, t2, false, late),
                );
                // The target named only in the MIDDLE of the body (e.g. step 20 of 40), so
                // neither a head nor a tail window shows it.
                let mid = generic / 2;
                push(
                    "diff-mid-step",
                    Expect::Different,
                    skill(
                        d.when_base.to_string(),
                        tier_steps(d, generic, t1, false, mid),
                    ),
                    d.when_base.to_string(),
                    tier_steps(d, generic, t2, false, mid),
                );
                // A shared ending: 8 identical steps after the target-naming one, so the tail
                // window is the same for both skills.
                let shared = generic.saturating_sub(8);
                push(
                    "diff-shared-ending",
                    Expect::Different,
                    skill(
                        d.when_base.to_string(),
                        tier_steps(d, generic, t1, false, shared),
                    ),
                    d.when_base.to_string(),
                    tier_steps(d, generic, t2, false, shared),
                );
            }
            (tier, cs)
        })
        .collect()
}

/// Cases the guard decides by CODE, asserted exactly offline: `existing` is never offered, so it
/// can never be refined — a blocked `same` case is an extra skill, a blocked `different` one is
/// correct, and when the proposal itself does not fit there is no judge call at all. Groups are
/// split with the production predicate (`fits_judge`), counted here, never hardcoded.
#[test]
fn guard_decides_blocked_cases_by_code() {
    let sets: Vec<(&str, Vec<Case>)> = [("single", cases()), ("multi", multi_cases())]
        .into_iter()
        .chain(length_tier_cases())
        .collect();
    for (set, cs) in &sets {
        let (blocked, judged): (Vec<&Case>, Vec<&Case>) = cs.iter().partition(|c| c.blocked());
        for c in &blocked {
            assert!(
                c.shortlist().iter().all(|s| s.key != c.existing.key),
                "{}: a skill the judge cannot see whole must never be offered",
                c.id
            );
            if !fits_judge(&c.proposal.when, &c.proposal.steps) {
                assert!(c.shortlist().is_empty(), "{}: no judge call", c.id);
            }
        }
        let s = score(
            &format!("guard {set} blocked"),
            &blocked.iter().map(|c| (*c).clone()).collect::<Vec<_>>(),
            |_| None,
        );
        assert_eq!(s.wrong_overwrite, 0);
        assert_eq!(
            s.extra_skill,
            blocked.iter().filter(|c| c.same()).count(),
            "every blocked same case becomes an extra skill, nothing else"
        );
        println!(
            "  {set:6} blocked={:?}\n  {set:6} judged ={:?}",
            blocked.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            judged.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        );
        // A set with nothing judged measures nothing live: fail instead of passing silently.
        // Every generated body (≤ ~3k chars) fits the 100k-unit field budget, so the blocked path
        // is exercised by the dedicated guard tests (fields and calls over budget), not here.
        assert!(!judged.is_empty(), "{set}: no case reaches the judge");
        assert!(
            blocked.is_empty(),
            "{set}: a generated case no longer fits the judge budget"
        );
    }
    // The 856-char distractor fits the 100k judge budget, but the 99-char proposal is only 0.116
    // of it, under `MIN_REFINE_LEN_RATIO` (0.15): the length floor drops it before the call, so
    // only `dich-tieng-anh` (ratio 0.943) reaches the judge.
    let c = multi_cases()
        .into_iter()
        .find(|c| c.id == "multi-same-short-beside-long")
        .unwrap();
    let offered: Vec<_> = c.shortlist().iter().map(|s| s.key.to_string()).collect();
    assert_eq!(offered, ["dich-tieng-anh"]);
}

/// Real store skills against the judge budget, from the production predicate. Reads the real
/// skills dir, read-only: `cargo test --bin aizen real_store_judge_budget -- --ignored --nocapture`
#[test]
#[ignore = "reads the real ~/.aizen skills dir"]
fn real_store_judge_budget() {
    let all = crate::skills::list();
    let fit = all.iter().filter(|s| fits_judge(&s.when, &s.body)).count();
    for s in &all {
        println!(
            "  {:40} when_units={:5} body_units={:6} budget={} body_chars={:6} fits_judge={}",
            s.key(),
            judge_field_units(&s.when),
            judge_field_units(&s.body),
            JUDGE_FIELD_UNITS,
            s.body.chars().count(),
            fits_judge(&s.when, &s.body)
        );
    }
    println!(
        "real store: {fit}/{} skills fit the judge budget",
        all.len()
    );
}

/// Acceptance thresholds for the S4 judge: no wrong overwrite in any set, and at most this many
/// extra skills per set and run. "Strict" was fixed before the head+tail clip run (fb67cae);
/// medium/long were relaxed to 4/12 after that run failed (7892a39, cargo exit 101), for data
/// safety, user-approved 2026-11-12, BEFORE any run of the whole-or-nothing guard. Changing them
/// after seeing results voids the measurement.
pub(crate) const MAX_EXTRA_SKILL: [(&str, usize); 7] = [
    ("single", 1),
    ("multi", 0),
    ("short", 0),
    ("medium", 4),
    ("long", 4),
    ("coverage", 1),
    ("holdout", 1),
];

// Set by the USER before the first live run of a non-empty blind set; never after a result.
pub(crate) const MAX_EXTRA_SKILL_BLIND: Option<usize> = None;

/// The held-out set is well formed, and every case reaches the judge: a case the shortlist or the
/// length floor never offers would score itself without testing the judge at all.
#[test]
fn holdout_cases_offline() {
    let cs = holdout_cases();
    assert_eq!(cs.len(), 10);
    assert_cases_well_formed(&cs, "holdout");
    for c in &cs {
        let offered = crate::persona::secretary::offer_candidates(
            &c.store(),
            &c.proposal.when,
            &c.proposal.steps,
        );
        assert!(
            offered.iter().any(|s| s.key == c.existing.key),
            "{}: existing skill must be offered to the judge",
            c.id
        );
    }
}
