//! The semantics `gameapp/test_pipeline.py` tests, as Rust tests. Each `// py:` comment names the Python check it carries over.
//! Not carried over (and why) is listed in docs/chain/PIPELINE-RUST.md: the checks that grep the Python source, that need the engine's own `settings.py` / `profile_guess.py`
//! from a clone (the base-document generator is an adapter here; its engine-dependent rows are the generator's tests), and `pyrewrite_driver`'s `sys.exit` handling.

mod common;

use common::*;
use ibuilder_pipeline::adapters::LauncherBase;
use ibuilder_pipeline::chain::{load_config, run_chain};
use ibuilder_pipeline::env::Env;
use ibuilder_pipeline::log::{MemLog, NullLog};
use ibuilder_pipeline::markers::*;
use ibuilder_pipeline::memory::{
    AddressSpace, FakeMem, SHADER_PER_JOB, rewrite_jobs, shader_jobs_for_memory,
};
use ibuilder_pipeline::profile::*;
use ibuilder_pipeline::rewrite::do_rewrite;
use ibuilder_pipeline::settings::*;
use ibuilder_pipeline::shaderpack::*;
use ibuilder_pipeline::shaders::{build_shaders_on_device, shader_parallelism};
use ibuilder_pipeline::testutil::set_mtime;
use ibuilder_pipeline::thermal::RealClock;
use ibuilder_pipeline::tsv::map_rows;
use ibuilder_pipeline::util::is_file;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const TID: &str = "0100DFA027504000";

// ---- the shader map the boot reads ---------------------------------------------------------------------------------------------------------

#[test]
fn shader_map_picks_this_games_map_not_the_first_id_in_sorted_order() {
    // py: "picks THIS game's map, not the first id in sorted order"
    let mut fx = Fx::new();
    fx.profile("# profile\ntitle_id\t0100BBBB00000000\nname\tB\n");
    write(
        fx.p("profile-derived/0100AAAA00000000/shader_map.tsv"),
        rows(3),
    );
    write(
        fx.p("profile-derived/0100BBBB00000000/shader_map.tsv"),
        rows(5),
    );
    set_mtime(
        fx.p("profile-derived/0100BBBB00000000/shader_map.tsv"),
        1_000_000,
    );
    let got = set_shader_map(&fx.ctx);
    assert!(
        got.to_string_lossy().contains("0100BBBB00000000"),
        "{got:?}"
    );
}

#[test]
fn shader_map_without_a_title_id_takes_the_most_recently_written() {
    // py: "no title_id in the profile: the most recently written map"
    let mut fx = Fx::new();
    fx.profile("# a profile with no title_id row\nname\tB\n");
    write(
        fx.p("profile-derived/0100AAAA00000000/shader_map.tsv"),
        rows(3),
    );
    write(
        fx.p("profile-derived/0100BBBB00000000/shader_map.tsv"),
        rows(5),
    );
    set_mtime(
        fx.p("profile-derived/0100AAAA00000000/shader_map.tsv"),
        1_000_000,
    );
    let got = set_shader_map(&fx.ctx);
    assert!(
        got.to_string_lossy().contains("0100BBBB00000000"),
        "{got:?}"
    );
}

#[test]
fn a_stale_empty_placeholder_does_not_shadow_a_real_derived_map() {
    // py: "a stale empty placeholder does not shadow a real derived map"
    let mut fx = Fx::new();
    fx.profile("title_id\t0100AAAA00000000\n");
    write(fx.p("shader_map.tsv"), rows(0));
    write(
        fx.p("profile-derived/0100AAAA00000000/shader_map.tsv"),
        rows(4),
    );
    let got = set_shader_map(&fx.ctx);
    assert!(
        got.to_string_lossy()
            .ends_with("0100AAAA00000000/shader_map.tsv"),
        "{got:?}"
    );
}

#[test]
fn only_empty_maps_one_of_them_is_used() {
    // py: "only empty maps exist: one of them is used"
    let mut fx = Fx::new();
    fx.profile("title_id\t0100AAAA00000000\n");
    write(fx.p("shader_map.tsv"), rows(0));
    write(
        fx.p("profile-derived/0100AAAA00000000/shader_map.tsv"),
        rows(0),
    );
    let got = set_shader_map(&fx.ctx);
    assert!(is_file(&got) && map_rows(&got) == 0, "{got:?}");
}

#[test]
fn nothing_at_all_a_header_only_map_is_written_and_named() {
    // py: "nothing at all: a header-only map is written (a missing file kills the boot)" + "and DM_NVN_SHMAP names it"
    let mut fx = Fx::new();
    fx.set_cfg("profile", &fx.p("profile.tsv").to_string_lossy());
    let got = set_shader_map(&fx.ctx);
    assert!(is_file(&got));
    assert_eq!(fs::read_to_string(&got).unwrap(), SHADER_MAP_HEADER);
    assert_eq!(
        fx.env.get("DM_NVN_SHMAP"),
        Some(got.to_string_lossy().into_owned())
    );
}

#[test]
fn an_env_shader_map_that_exists_is_left_alone() {
    let mut fx = Fx::new();
    fx.profile("title_id\t0100AAAA00000000\n");
    write(fx.p("mine.tsv"), rows(2));
    fx.env
        .set("DM_NVN_SHMAP", &fx.p("mine.tsv").to_string_lossy());
    assert_eq!(set_shader_map(&fx.ctx), fx.p("mine.tsv"));
    assert!(fx.log.contains("DM_NVN_SHMAP was already set"));
}

// ---- the boot-only request, through the chain -------------------------------------------------------------------------------------------------------

struct Scenario {
    fx: Fx,
    boot: Arc<FakeBoot>,
    req: PathBuf,
    died: bool,
    /// the chain's result code (`None` when the run died)
    rc: Option<i32>,
}

fn scenario(
    mode: &str,
    make_framework: bool,
    config_build: Option<&str>,
    stamp: Option<&str>,
    before: Option<&dyn Fn(&Fx)>,
) -> Scenario {
    let mut fx = Fx::new();
    let docs = fx.docs().to_path_buf();
    let extract = docs.join("extract");
    let fw = docs.join("fw/DmGameProbe.framework/DmGameProbe");
    fs::create_dir_all(extract.join("flat")).unwrap();
    if make_framework {
        write(&fw, "x");
    }
    let cfgp = docs.join("gameapp.config");
    write(
        &cfgp,
        format!(
            "extract_out={}\nframework_out={}\nprofile={}\nromfs=\n{}",
            extract.display(),
            fw.display(),
            docs.join("profile.tsv").display(),
            config_build
                .map(|b| format!("app_build={b}\n"))
                .unwrap_or_default()
        ),
    );
    if let Some(s) = stamp {
        write(docs.join("gameapp.framework"), s);
    }
    let req = docs.join("gameapp.boot-only");
    write(
        &req,
        "boot the framework that is already built and signed\n",
    );
    write(docs.join("gameapp.running"), "marker");
    if let Some(b) = before {
        b(&fx);
    }
    let boot = FakeBoot::new(req.clone(), mode);
    fx.ctx.adapters.boot = Some(boot.clone());
    fx.ctx.boot_enabled = true;
    load_config(&mut fx.ctx, Some(&cfgp));
    let rep = catch_unwind(AssertUnwindSafe(|| run_chain(&mut fx.ctx)));
    let died = rep.is_err();
    Scenario {
        fx,
        boot,
        req,
        died,
        rc: rep.ok().map(|r| r.rc),
    }
}

const BUILD: &str = "20261117-173055-c842913";

#[test]
fn boot_returns_the_request_is_attempted_during_the_boot_and_consumed_after() {
    // py: "boot returns: the request is attempted DURING the boot, not removed before it" + "consumed afterwards"
    let s = scenario("ok", true, None, None, None);
    assert!(s.boot.seen().request.unwrap().starts_with("attempted"));
    assert!(!s.req.exists());
}

#[test]
fn a_process_that_dies_inside_the_boot_leaves_the_request_attempted_and_the_marker() {
    // py: "the process dies inside the boot (simulated)" + "the request is still on disk" + "and says attempted" + "the interrupted-run marker is still there"
    let s = scenario("die", true, None, None, None);
    assert!(s.died);
    assert!(
        s.req.exists(),
        "it evaporated: the next launch has nothing to resume"
    );
    assert!(fs::read_to_string(&s.req).unwrap().starts_with("attempted"));
    assert!(s.fx.p("gameapp.running").exists());
}

#[test]
fn dlopen_failure_stage3_keeps_the_request_attempted() {
    // py: "dlopen fails (stage 3): the request stays, attempted"
    let s = scenario("stage3", true, None, None, None);
    assert!(s.req.exists() && fs::read_to_string(&s.req).unwrap().starts_with("attempted"));
}

#[test]
fn a_boot_only_launch_whose_boot_did_not_run_does_not_return_0() {
    // review r3 §5.2: 0 is "the boot ran"; a dlopen that failed (stage 3) is a failed stage 5, and 3 is kept for "nothing to boot"
    let s = scenario("stage3", true, None, None, None);
    assert_eq!(
        s.rc,
        Some(ibuilder_pipeline::chain::RC_FAILED),
        "{:?}",
        s.fx.log.lines()
    );
    let s = scenario("ok", true, None, None, None);
    assert_eq!(s.rc, Some(0));
    let s = scenario("ok", false, None, None, None);
    assert_eq!(s.rc, Some(ibuilder_pipeline::chain::RC_NOTHING_TO_BOOT));
}

#[test]
fn no_framework_on_disk_the_request_is_consumed() {
    // py: "no framework on disk: the request is consumed (it can never be honoured)"
    let s = scenario("ok", false, None, None, None);
    assert!(!s.req.exists());
    assert_eq!(s.boot.calls(), 0);
}

#[test]
fn only_a_framework_this_build_made_is_booted_without_a_rebuild() {
    // py: "a framework this build made is booted" + "and its request is consumed afterwards"
    let s = scenario(
        "ok",
        true,
        Some(BUILD),
        Some(&format!("app_build={BUILD}\ngame=Grizzy.nsp\n")),
        None,
    );
    assert_eq!(s.boot.calls(), 1);
    assert!(!s.req.exists());
    // py: "a framework ANOTHER build made is not booted" + "and the request that pointed at it is consumed"
    let s = scenario(
        "ok",
        true,
        Some(BUILD),
        Some("app_build=20261116-100000-oldoldo\ngame=Grizzy.nsp\n"),
        None,
    );
    assert_eq!(s.boot.calls(), 0);
    assert!(!s.req.exists());
    // py: "a framework with no stamp (built before builds were tracked) is not booted"
    let s = scenario("ok", true, Some(BUILD), None, None);
    assert_eq!(s.boot.calls(), 0);
    // py: "a config that names no app_build skips the check (a harness that predates it)"
    let s = scenario("ok", true, None, None, None);
    assert_eq!(s.boot.calls(), 1);
}

#[test]
fn an_app_update_with_the_same_framework_id_boots_the_framework_already_built() {
    // the framework id (scripts/lib-hash.sh framework_id_listing): only what shapes a built framework; a new app build id alone is not a rebuild
    let fresh = |config: &str, stamp: &str| {
        scenario("ok", true, Some(config), Some(stamp), None)
            .boot
            .calls()
    };
    let stamp = "app_build=20261120-100000-rustaaaa\nframework_id=F1\ngame=Grizzy.nsp\n";
    assert_eq!(
        fresh("20261122-090000-rustbbbb\nframework_id=F1", stamp),
        1,
        "same framework id, another build id: booted, no rebuild"
    );
    let s = scenario(
        "ok",
        true,
        Some("20261120-100000-rustaaaa\nframework_id=F2"),
        Some(stamp),
        None,
    );
    assert_eq!(
        s.boot.calls(),
        0,
        "another framework id is a rebuild, even with the same build id"
    );
    assert!(!s.req.exists(), "and the request is consumed");
    assert!(
        s.fx.log
            .lines()
            .iter()
            .any(|l| l.contains("(framework id F1)") && l.contains("(framework id F2)")),
        "{:?}",
        s.fx.log.lines()
    );
    // a stamp from before framework ids: the build id decides, as before
    assert_eq!(
        fresh(
            &format!("{BUILD}\nframework_id=F1"),
            &format!("app_build={BUILD}\ngame=G\n")
        ),
        1
    );
    assert_eq!(
        fresh(
            "20261122-090000-rustbbbb\nframework_id=F1",
            &format!("app_build={BUILD}\ngame=G\n")
        ),
        0
    );
}

// ---- the save and temp folders the engine aborts without (DM_SAVE) ---------------------------------------------------------------------------------------

#[test]
fn the_boot_sees_dm_save_and_dm_temp_and_the_folders_exist() {
    // py: "the boot sees DM_SAVE" + "and the folder exists" + "DM_TEMP too"
    let s = scenario(
        "ok",
        true,
        Some(BUILD),
        Some(&format!("app_build={BUILD}\n")),
        None,
    );
    let seen = s.boot.seen();
    let save = seen.save.expect("DM_SAVE at the boot");
    assert!(Path::new(&save).is_dir());
    assert!(Path::new(&seen.temp.expect("DM_TEMP at the boot")).is_dir());
}

#[test]
fn a_profile_that_names_its_title_gets_a_per_game_folder() {
    // py: "a profile that names its title gets a per-game folder (two games never share a save)" + "both created" + "and said in the log"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    set_game_env(&fx.ctx);
    let want = fx.p(&format!("games/{TID}"));
    assert_eq!(
        fx.env.get("DM_SAVE"),
        Some(want.join("save").to_string_lossy().into_owned())
    );
    assert_eq!(
        fx.env.get("DM_TEMP"),
        Some(want.join("temp").to_string_lossy().into_owned())
    );
    assert!(want.join("save").is_dir() && want.join("temp").is_dir());
    assert!(fx.log.contains("DM_SAVE="));
}

#[test]
fn no_title_in_the_profile_documents_save_and_temp() {
    // py: "no title in the profile: Documents/save and Documents/temp"
    let mut fx = Fx::new();
    fx.set_cfg("profile", &fx.p("no-such-profile.tsv").to_string_lossy());
    set_game_env(&fx.ctx);
    assert_eq!(
        fx.env.get("DM_SAVE"),
        Some(fx.p("save").to_string_lossy().into_owned())
    );
    assert!(fx.p("save").is_dir());
}

#[test]
fn the_bundled_font_stand_ins_are_handed_to_the_host_and_an_existing_choice_wins() {
    // the engine's "Shared fonts on iOS": a game that waits for GetSharedFontLoadState never boots without DM_FONT_DIR (Mario Kart 8 Deluxe, 2026-11-19)
    let fx = Fx::new();
    set_game_env(&fx.ctx);
    assert!(
        !fx.env.truthy("DM_FONT_DIR"),
        "no dm/font in the bundle: nothing is set"
    );
    let fonts = fx.ctx.bundle.join("dm").join("font");
    write(fonts.join("standard.ttf"), "font");
    set_game_env(&fx.ctx);
    assert_eq!(
        fx.env.get("DM_FONT_DIR"),
        Some(fonts.to_string_lossy().into_owned())
    );
    assert!(fx.log.contains("DM_FONT_DIR="));
    let again = Fx::new();
    write(again.ctx.bundle.join("dm/font/standard.ttf"), "font");
    again.env.set("DM_FONT_DIR", "/elsewhere/fonts");
    set_game_env(&again.ctx);
    assert_eq!(
        again.env.get("DM_FONT_DIR").as_deref(),
        Some("/elsewhere/fonts")
    );
}

#[test]
fn a_draw_waits_for_its_pipeline_unless_the_player_or_the_config_chose_otherwise() {
    // a first race drew only the HUD while 112 libraries compiled in the background: the host's default (skip the draw) is not the default here
    let fx = Fx::new();
    set_game_env(&fx.ctx);
    assert_eq!(fx.env.get("DM_PSO_NOWAIT").as_deref(), Some("0"));
    let chosen = Fx::new();
    chosen.env.set("DM_PSO_NOWAIT", "1");
    set_game_env(&chosen.ctx);
    assert_eq!(
        chosen.env.get("DM_PSO_NOWAIT").as_deref(),
        Some("1"),
        "an env.txt / harness choice wins"
    );
    let mut cfg = Fx::new();
    cfg.ctx.cfg.set("pso_nowait", "1");
    set_game_env(&cfg.ctx);
    assert!(
        !cfg.env.truthy("DM_PSO_NOWAIT"),
        "pso_nowait=1 leaves the host's own policy"
    );
}

#[test]
fn a_dm_save_already_in_the_environment_wins() {
    // py: "a DM_SAVE already in the environment wins" + "and does not stop DM_TEMP from being made"
    let fx = Fx::new();
    fx.env.set("DM_SAVE", "/elsewhere/save");
    set_game_env(&fx.ctx);
    assert_eq!(fx.env.get("DM_SAVE").as_deref(), Some("/elsewhere/save"));
    assert!(fx.log.contains("already set"));
    assert!(fx.env.truthy("DM_TEMP"));
}

#[test]
fn a_folder_that_cannot_be_made_is_said_with_its_consequence() {
    // py: "a folder that cannot be made is said, with its consequence, and DM_SAVE is left unset"
    let mut fx = Fx::new();
    write(fx.p("games"), "a file where the folder should go");
    fx.profile(&format!("title_id\t{TID}\n"));
    set_game_env(&fx.ctx);
    assert_eq!(fx.env.get("DM_SAVE"), None);
    assert!(
        fx.log
            .any(|l| l.contains("could not create") && l.contains("abort")),
        "{:?}",
        fx.log.lines()
    );
}

// ---- the shader pack ---------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn an_imported_pack_is_installed_and_the_host_pointed_at_it() {
    // py: "an imported pack is installed into Documents/shaders/<id>/", "and its .tar removed", "the host is pointed at its shader map", "and at its library folder",
    //     "the log names the shaders and the chain, and says it is the host's", "the shader map step keeps the pack's map"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("shader_chain", "19d183a2b462");
    make_pack(
        &fx.p(&format!("{TID}.shaders.tar")),
        TID,
        &default_pack_files(),
    );
    let got = set_shader_pack(&fx.ctx);
    let pack = fx.p(&format!("shaders/{TID}"));
    assert_eq!(got.as_deref(), Some(pack.as_path()));
    assert!(is_file(&pack.join("b-19d183a2b462.img")));
    assert!(
        !fx.p(&format!("{TID}.shaders.tar")).exists(),
        "the folder is the pack now: no second copy of ~2 GB"
    );
    assert_eq!(
        fx.env.get("DM_NVN_SHMAP"),
        Some(pack.join("shader_map.tsv").to_string_lossy().into_owned())
    );
    assert_eq!(
        fx.env.get("DM_MSL_DIR"),
        Some(pack.to_string_lossy().into_owned()),
        "the host reads <DM_MSL_DIR>/b-<chain>.img"
    );
    assert!(
        fx.log.any(|l| l.contains("1 shaders")
            && l.contains("b-19d183a2b462")
            && l.contains("host's chain")),
        "{:?}",
        fx.log.lines()
    );
    assert_eq!(set_shader_map(&fx.ctx), pack.join("shader_map.tsv"));
}

#[test]
fn a_pack_of_another_chain_is_installed_but_said_to_be_refused_with_the_fix() {
    // py: "a pack of another chain is installed but said to be refused by this host, with the fix"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("shader_chain", "19d183a2b462");
    make_pack(
        &fx.p("old.shaders.tar"),
        TID,
        &[
            (
                "shader_map.tsv",
                "package\tindex\tstage\tdm_ctl_sha16\tsource\nmain\t0\tvs\tabcdef0123456789\tx\n",
            ),
            ("b-d24f05cf9d78.img", "IMG"),
        ],
    );
    set_shader_pack(&fx.ctx);
    assert!(
        fx.log.any(|l| l.contains("[RED]")
            && l.contains("d24f05cf9d78")
            && l.contains("19d183a2b462")
            && l.contains("make the pack again")),
        "{:?}",
        fx.log.lines()
    );
}

#[test]
fn a_pack_with_a_member_outside_its_folder_is_refused_whole_and_nothing_escapes() {
    // py: "a pack with a member outside its folder is refused whole, and nothing escapes" + "and it stays where it is"
    let fx = Fx::new();
    let mut files = default_pack_files();
    files.push(("../../escape.txt", "x"));
    make_pack(&fx.p("evil.shaders.tar"), TID, &files);
    install_shader_packs(&fx.ctx);
    assert!(!fx.p("escape.txt").exists() && !fx.p(&format!("shaders/{TID}")).is_dir());
    assert!(fx.log.contains("refused"));
    assert!(
        fx.p("evil.shaders.tar").exists(),
        "nothing is deleted on a refusal"
    );
}

#[test]
fn no_pack_nothing_is_set_and_the_log_says_stage_3_builds_it_here() {
    // py: "no pack: nothing is set, and the log says the game will draw nothing and that stage 3 builds it here"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    assert!(set_shader_pack(&fx.ctx).is_none());
    assert!(!fx.env.truthy("DM_MSL_DIR"));
    assert!(fx.log.any(|l| l.contains("translates them on this device")
        && l.contains(&format!("{TID}.shaders.tar"))));
}

#[test]
fn a_boot_with_a_pack_imported_the_host_sees_the_library_folder_and_map() {
    // py: "a boot with a pack imported: the host sees the pack's library folder and its shader map"
    let before = |fx: &Fx| {
        write(fx.p("profile.tsv"), format!("title_id\t{TID}\n"));
        make_pack(
            &fx.p(&format!("{TID}.shaders.tar")),
            TID,
            &default_pack_files(),
        );
    };
    let s = scenario(
        "ok",
        true,
        Some(BUILD),
        Some(&format!("app_build={BUILD}\n")),
        Some(&before),
    );
    let seen = s.boot.seen();
    let pack = s.fx.p(&format!("shaders/{TID}"));
    assert_eq!(seen.msl, Some(pack.to_string_lossy().into_owned()));
    assert_eq!(
        seen.shmap,
        Some(pack.join("shader_map.tsv").to_string_lossy().into_owned())
    );
}

#[test]
fn a_pack_whose_folder_is_not_a_title_id_is_refused_and_no_other_pack_is_touched() {
    // py: "a pack whose folder is not a title id ('.') is refused, and no other game's pack is touched"
    let fx = Fx::new();
    make_pack(&fx.p("dot.shaders.tar"), ".", &default_pack_files());
    write(fx.p(&format!("shaders/{TID}/shader_map.tsv")), "x");
    install_shader_packs(&fx.ctx);
    assert!(is_file(&fx.p(&format!("shaders/{TID}/shader_map.tsv"))));
    assert!(fx.log.contains("not a title id"), "{:?}", fx.log.lines());
}

// ---- the heap check from the phone's own last boot, and the player's game settings --------------------------------------------------------------------------

const SP: &str = "010043600B6A6000";

fn heap_fixture() -> Fx {
    let mut fx = Fx::new();
    write(
        fx.p(&format!("profile-derived/{SP}/game.tsv")),
        format!(
            "title_id\t{SP}\nmem_reserve\t0x140000000\nmem_alias\t0x118000000\nmem_stack\t0x10000000\napp_pool\t0xCC800000\nsys_resource\t0x1000000\n"
        ),
    );
    fx.profile(&format!("title_id\t{SP}\n"));
    write(
        fx.p("gameapp.guest.log"),
        format!(
            "[dm] build: commit not recorded, shader chain x, profile 0100DFA027504000\n[dm] SetHeapSize OOM: req 0x10000000 > cap 0x8000000\n[dm] build: commit not recorded, shader chain x, profile {SP}\n[dm] >>> nninitStartup\n[dm] SetHeapSize OOM: req 0xc9400000 > cap 0x8000000\n"
        ),
    );
    fx
}

#[test]
fn the_heap_check_moves_the_alias_split_for_a_3_15_gb_heap() {
    // py: "the last boot of THIS game that asked for a 3.15 GB heap moves the alias / heap split", "under a note that says why", "and the log says it",
    //     "a second look at the same boot changes nothing", "a boot with no signal changes nothing"
    let mut fx = heap_fixture();
    let prof = Arc::new(RecordingProfile::default());
    fx.ctx.adapters.profile = prof.clone();
    let got = apply_heap_check(&fx.ctx);
    let gt = fs::read_to_string(fx.p(&format!("profile-derived/{SP}/game.tsv"))).unwrap();
    assert_eq!(
        got.get("mem_alias").map(String::as_str),
        Some("0x50000000"),
        "{got:?}"
    );
    assert!(gt.contains("mem_alias\t0x50000000") && !gt.contains("mem_alias\t0x118000000"));
    assert!(gt.contains("# heap check (this device): mem_alias 0x50000000"));
    assert!(fx.log.contains("heap check: mem_alias 0x50000000"));
    assert_eq!(
        prof.renders.lock().unwrap().len(),
        1,
        "profile.tsv is rendered again"
    );
    assert!(prof.renders.lock().unwrap()[0].ends_with(&format!(
        "DM_GAMES_DIR={}",
        fx.p("profile-derived").display()
    )));
    assert!(
        apply_heap_check(&fx.ctx).is_empty(),
        "the profile already follows it"
    );
    write(
        fx.p("gameapp.guest.log"),
        format!("[dm] build: commit not recorded, shader chain x, profile {SP}\n[dm] >>> nnMain\n"),
    );
    assert!(apply_heap_check(&fx.ctx).is_empty());
}

#[test]
fn a_failing_render_is_logged_red_and_the_table_keeps_the_keys() {
    let mut fx = heap_fixture();
    fx.ctx.adapters.profile = Arc::new(RecordingProfile {
        fail_render: true,
        ..Default::default()
    });
    let got = apply_heap_check(&fx.ctx);
    assert_eq!(got.len(), 1);
    assert!(
        fx.log
            .contains("[RED] the corrected profile could not be rendered")
    );
}

#[test]
fn the_players_settings_become_the_engines_switches_before_the_boot() {
    // py: "the screen's settings become the engine's switches before the boot", "and are said in the log", "auto leaves the profile's choice", "an environment that already names a switch wins"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    write(
        fx.p(&format!("games/{TID}/settings.tsv")),
        "mode\thandheld\nfps\t30\npicture\tfill\nshow_fps\t0\nbogus\tx\n",
    );
    let got = apply_game_settings(&fx.ctx);
    let want: std::collections::BTreeMap<String, String> = [
        ("DM_OPMODE", "0"),
        ("DM_FPS_TARGET", "30"),
        ("DM_PICTURE_FIT", "fill"),
        ("DM_SHOW_FPS", "0"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();
    assert_eq!(got, want);
    assert_eq!(fx.env.get("DM_OPMODE").as_deref(), Some("0"));
    assert!(fx.log.contains("DM_FPS_TARGET=30"));
    for v in [
        "DM_OPMODE",
        "DM_FPS_TARGET",
        "DM_PICTURE_FIT",
        "DM_SHOW_FPS",
    ] {
        fx.env.remove(v);
    }
    write(
        fx.p(&format!("games/{TID}/settings.tsv")),
        "mode\tauto\nfps\tauto\n",
    );
    assert!(
        apply_game_settings(&fx.ctx).is_empty(),
        "auto leaves the profile's choice"
    );
    fx.env.set("DM_FPS_TARGET", "60");
    write(fx.p(&format!("games/{TID}/settings.tsv")), "fps\t30\n");
    assert!(apply_game_settings(&fx.ctx).is_empty());
    assert_eq!(fx.env.get("DM_FPS_TARGET").as_deref(), Some("60"));
}

// ---- review fixes: a new chain starts clean, the profile's modules, registered games' tables --------------------------------------------------------------------

#[test]
fn a_new_chain_starts_clean() {
    // py: "a new chain removes the previous framework stamp first", "and the previous rewrite and flat folders", "and NSZ leftovers of a killed launch"
    let mut fx = Fx::new();
    let ex = fx.p("extract");
    fs::create_dir_all(ex.join("rewrite/subsdk0")).unwrap();
    fs::create_dir_all(ex.join("flat")).unwrap();
    write(fx.p("gameapp.framework"), "app_build=x\ngame=A.nsp\n");
    write(fx.p(".nsz/B.nsp.part"), "x");
    fx.set_cfg("extract_out", &ex.to_string_lossy());
    invalidate_previous_build(&fx.ctx);
    assert!(
        !fx.p("gameapp.framework").exists(),
        "no Ready for the old game over the new files"
    );
    assert!(!ex.join("rewrite").exists() && !ex.join("flat").exists());
    assert!(fs::read_dir(fx.p(".nsz")).unwrap().next().is_none());
}

#[test]
fn registered_games_tables_and_the_profiles_modules() {
    // py: "a registered game's table is found in the bundle", "the modules come from the game's profile, not the config's fixed list", "a table this app derived wins over the bundled one"
    let mut fx = Fx::new();
    let bundle = TempBundle::new(&fx);
    fx.ctx.bundle = bundle.path.clone();
    let tid = "0100152000022000";
    write(
        bundle.path.join(format!("dm/games/{tid}/game.tsv")),
        format!("title_id\t{tid}\nmodules\tmain subsdk0 subsdk1\n"),
    );
    fx.profile(&format!("title_id\t{tid}\n"));
    assert_eq!(
        table_dir(&fx.ctx, tid),
        Some(bundle.path.join(format!("dm/games/{tid}")))
    );
    assert_eq!(
        profile_modules(&fx.ctx).as_deref(),
        Some("main subsdk0 subsdk1")
    );
    write(
        fx.p(&format!("profile-derived/{tid}/game.tsv")),
        format!("title_id\t{tid}\nmodules\tmain sdk\n"),
    );
    assert_eq!(profile_modules(&fx.ctx).as_deref(), Some("main sdk"));
}

struct TempBundle {
    path: PathBuf,
}
impl TempBundle {
    fn new(fx: &Fx) -> TempBundle {
        let path = fx.p("bundle");
        fs::create_dir_all(&path).unwrap();
        TempBundle { path }
    }
}

// ---- the stamp itself ------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn the_stamp_records_the_build_and_the_game() {
    // py: "the stamp records the build and the game it was made for", "and when it was built", "a chain-only run keeps the game already stamped", "a new extraction replaces the game"
    let mut fx = Fx::new();
    fx.cfg(&format!("app_build={BUILD}\ngame=Grizzy.nsp"));
    write_framework_stamp(&fx.ctx);
    let got = read_framework_stamp(&fx.ctx);
    assert_eq!(got.get("app_build"), Some(BUILD));
    assert_eq!(got.get("game"), Some("Grizzy.nsp"));
    assert!(got.get_or("built", "").ends_with('Z'));
    fx.cfg(&format!("app_build={BUILD}"));
    write_framework_stamp(&fx.ctx);
    assert_eq!(
        read_framework_stamp(&fx.ctx).get("game"),
        Some("Grizzy.nsp")
    );
    fx.cfg(&format!("app_build={BUILD}\ngame=Other.nsp"));
    write_framework_stamp(&fx.ctx);
    assert_eq!(read_framework_stamp(&fx.ctx).get("game"), Some("Other.nsp"));
}

#[test]
fn the_stamp_records_the_framework_id_beside_the_build() {
    let mut fx = Fx::new();
    fx.cfg(&format!(
        "app_build={BUILD}\nframework_id=0123456789abcdef\ngame=Grizzy.nsp"
    ));
    write_framework_stamp(&fx.ctx);
    let text = fs::read_to_string(fx.p("gameapp.framework")).unwrap();
    assert!(
        text.starts_with(&format!(
            "app_build={BUILD}\nframework_id=0123456789abcdef\ngame=Grizzy.nsp\nbuilt="
        )),
        "{text}"
    );
    assert!(fx.log.contains("framework_id=0123456789abcdef"));
    fx.cfg(&format!("app_build={BUILD}\ngame=Grizzy.nsp"));
    write_framework_stamp(&fx.ctx);
    assert!(
        !fs::read_to_string(fx.p("gameapp.framework"))
            .unwrap()
            .contains("framework_id"),
        "a config with no framework id stamps the four lines it always did"
    );
}

#[test]
fn no_app_build_nothing_is_stamped() {
    // py: "no app_build in the config: nothing is stamped, and it says so"
    let mut fx = Fx::new();
    fx.cfg("game=Grizzy.nsp");
    write_framework_stamp(&fx.ctx);
    assert!(!fx.p("gameapp.framework").exists());
    assert!(fx.log.contains("no app_build"));
}

// ---- pulling libraries only when the profile turns Metal 4 on ---------------------------------------------------------------------------------------------------------

#[test]
fn pulling_libraries_only_when_the_profile_turns_metal4_on() {
    // py: the five "pulling" checks
    let fx = Fx::new();
    let prof = fx.p("prof");
    fs::create_dir_all(&prof).unwrap();
    let cfg = |s: &str| ibuilder_pipeline::config::Config::parse(s);
    assert!(
        !shader_pulling_wanted(&prof, &cfg("")),
        "no profile: no pulling libraries"
    );
    write(
        prof.join("game.tsv"),
        "modules\tmain sdk\npso_mtl4\ton   # BotW\n",
    );
    assert!(shader_pulling_wanted(&prof, &cfg("")));
    assert!(
        !shader_pulling_wanted(&prof, &cfg("shader_pull=0")),
        "unless the config forces it off"
    );
    write(prof.join("game.tsv"), "pso_mtl4\toff\n");
    assert!(!shader_pulling_wanted(&prof, &cfg("")));
    assert!(
        shader_pulling_wanted(&prof, &cfg("shader_pull=1")),
        "unless the config forces it on"
    );
}

// ---- the rewrite cache -------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn the_rewrite_cache() {
    // py: the rewrite cache block: first run, same inputs from the cache with the same output, changed profile, new app build, another game version, no app_build
    let mut fx = Fx::new();
    let rw = Arc::new(CountingRewriter::default());
    fx.ctx.adapters.rewriter = rw.clone();
    let tid = "0100ABCD12340000";
    fx.profile(&format!("title_id\t{tid}\n"));
    write(
        fx.p(&format!("profile-derived/{tid}/game.tsv")),
        "modules\tmain\nheap\t1\n",
    );
    write(fx.p("exe/main"), "MAIN-1");
    fx.cfg("");
    fx.profile(&format!("title_id\t{tid}\n"));
    fx.set_cfg("exe_dir", &fx.p("exe").to_string_lossy());
    fx.set_cfg("modules", "main");
    fx.set_cfg("extract_out", &fx.p("extract").to_string_lossy());
    fx.set_cfg("app_build", "B1");
    let runs = |rw: &CountingRewriter| rw.runs.lock().unwrap().len();
    let out = do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 1, "first rewrite runs the rewriter");
    fs::remove_dir_all(&out).unwrap(); // what invalidate_previous_build does at the next chain
    fx.log.clear();
    do_rewrite(&mut fx.ctx).unwrap();
    assert!(
        runs(&rw) == 1 && fx.log.contains("from the cache"),
        "{:?}",
        fx.log.lines()
    );
    assert_eq!(
        fs::read(out.join("main/main.patched.bin")).unwrap(),
        b"MAIN-1.patched.bin",
        "with the same output"
    );
    write(
        fx.p(&format!("profile-derived/{tid}/game.tsv")),
        "modules\tmain\nheap\t2\n",
    ); // the heap check edits it
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 2, "a changed profile rewrites again");
    fx.set_cfg("app_build", "B2");
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 3, "a new app build rewrites again");
    write(fx.p("exe/main"), "MAIN-2");
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 4);
    assert_eq!(
        fs::read(out.join("main/main.patched.bin")).unwrap(),
        b"MAIN-2.patched.bin",
        "another game version rewrites again"
    );
    fx.ctx.cfg.remove("app_build");
    do_rewrite(&mut fx.ctx).unwrap();
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 6, "no app_build: never cached");
    // with a framework id the cache belongs to it, not to the app build: an app update that keeps the framework id keeps the rewrite cache
    fx.set_cfg("app_build", "B3");
    fx.set_cfg("framework_id", "F1");
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(
        runs(&rw),
        7,
        "a cache made under an app build is not taken for a framework id"
    );
    fx.set_cfg("app_build", "B4");
    do_rewrite(&mut fx.ctx).unwrap();
    assert!(
        runs(&rw) == 7 && fx.log.contains("from the cache"),
        "same framework id, another app build: from the cache"
    );
    assert_eq!(
        fs::read(out.join("main/main.patched.bin")).unwrap(),
        b"MAIN-2.patched.bin"
    );
    fx.set_cfg("framework_id", "F2");
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 8, "another framework id rewrites again");
    fx.ctx.cfg.remove("app_build");
    fx.set_cfg("framework_id", "F2");
    do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(runs(&rw), 8, "a framework id alone is enough to cache");
}

// ---- a library of another chain is removed before stage 3 builds ------------------------------------------------------------------------------------------------------

#[test]
fn a_library_of_another_chain_is_removed_before_the_build() {
    // py: "a library of another chain is removed before the build"
    let mut fx = Fx::new();
    let pack = fx.p(&format!("shaders/{TID}"));
    write(pack.join("b-19d183a2b462.img"), "old library");
    write(pack.join("shader_map.tsv"), rows(3));
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("shader_chain", "cc0f2129e1de");
    build_shaders_on_device(&mut fx.ctx);
    assert!(!pack.join("b-19d183a2b462.img").exists());
    assert!(
        fx.log.contains("removed (chain 19d183a2b462"),
        "{:?}",
        fx.log.lines()
    );
}

// ---- the game's own settings screen (post-processing of the engine's rows) --------------------------------------------------------------------------------------------

struct Rows;
impl LauncherBase for Rows {
    fn base_document(
        &self,
        _v: &ibuilder_pipeline::View<'_>,
        tid: &str,
        _m: &Path,
        _c: &Path,
        _g: &str,
        _d: Option<&Path>,
    ) -> ibuilder_pipeline::Result<serde_json::Value> {
        let t = |env: &str, default: &str| serde_json::json!({"env": env, "kind": "toggle", "section": "Game", "label": env, "note": "", "default": default, "on": "1", "off": "0"});
        let mut d = settings_doc(tid);
        d["settings"] = serde_json::json!([
            serde_json::json!({"env": "DM_FPS_TARGET", "kind": "choice", "section": "Game", "label": "Frame rate", "note": "", "default": "30", "options": [["30", "30 fps"], ["60", "60 fps"]]}),
            serde_json::json!({"env": "DM_OPMODE", "kind": "choice", "section": "Game", "label": "Console mode", "note": "short", "default": "1", "options": [["1", "Docked"], ["0", "Handheld"]]}),
            t("DM_DYNAMIC_RESOLUTION", "0"),
            serde_json::json!({"env": "DM_LANG", "kind": "choice", "section": "Game", "label": "Game language", "note": "", "default": "en", "options": [["en", "English"], ["fr", "French"]]}),
            t("DM_PSO_MTL4", "1"),
            t("DM_PSO_PREBUILD", "1"),
            t("DM_SHOW_FPS", "0"),
            t("DM_NVN_PREFETCH", "1"),
            t("DM_FRAMEGEN", "0")
        ]);
        Ok(d)
    }
}
fn settings_doc(tid: &str) -> serde_json::Value {
    ibuilder_pipeline::settings::empty_document(tid)
}

fn launcher_fixture() -> (Fx, serde_json::Value, PathBuf) {
    let mut fx = Fx::new();
    fx.ctx.adapters.launcher = Some(Arc::new(Rows));
    let tid = "0100B8E016F76000";
    fx.profile(&format!("title_id\t{tid}\n"));
    fx.set_cfg("extract_out", &fx.p("extract").to_string_lossy());
    let patched = fx.p("extract/rewrite");
    write(
        patched.join("main/main.patched.bin"),
        b"x\0nvnBootstrapLoader\0y",
    );
    write(
        patched.join("main/main.fixups.tsv"),
        "# DYNAMIC_RESOLUTION\t1\n# RENDER_SCALE\t1\n",
    );
    let out =
        write_launcher_settings(&fx.ctx, &patched.join("main")).expect("launcher.json written");
    let doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
    (fx, doc, out)
}

#[test]
fn launcher_json_rows_notes_and_gameapp_additions() {
    // py: "launcher.json lists the game's own rows", "but not the Metal 4 rows its pack cannot serve", "every row the player sees has a line of explanation", "the console mode says it is how to lower any game's resolution",
    //     "plus GameApp's Low latency row", "the counter defaults to on, as the app shows it", gfxapi nvn
    let (_fx, doc, _out) = launcher_fixture();
    let envs: Vec<String> = doc["settings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["env"].as_str().unwrap().to_string())
        .collect();
    assert!(
        envs.contains(&"DM_FPS_TARGET".to_string()) && envs.contains(&"DM_OPMODE".to_string()),
        "{envs:?}"
    );
    assert!(
        !envs.contains(&"DM_PSO_MTL4".to_string())
            && !envs.contains(&"DM_PSO_PREBUILD".to_string()),
        "the Metal 4 rows its pack cannot serve"
    );
    assert!(
        envs.contains(&"DM_PRESENT_DRAIN".to_string())
            && envs.contains(&"DM_LIVE_SHADERS".to_string()),
        "GameApp's own rows for an NVN game"
    );
    assert_eq!(doc["gfxapi"], "nvn");
    let notes: Vec<(String, String)> = doc["settings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["env"].as_str().unwrap().to_string(),
                r["note"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();
    let missing: Vec<&str> = notes
        .iter()
        .filter(|(e, n)| {
            n.is_empty() && e != "DM_DYNAMIC_RESOLUTION" && e != "DM_LANG" && e != "DM_NVN_PREFETCH"
        })
        .map(|(e, _)| e.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "every row with a catalogue line has one: {missing:?}"
    );
    let note = |e: &str| notes.iter().find(|(k, _)| k == e).unwrap().1.clone();
    assert!(
        note("DM_OPMODE").contains("lower the resolution"),
        "the replacement note wins over the catalogue's short one"
    );
    assert!(note("DM_FRAMEGEN").contains("120"));
    let show_fps = doc["settings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["env"] == "DM_SHOW_FPS")
        .unwrap();
    assert_eq!(show_fps["default"], "1");
    assert!(doc["catalogue"].as_str().unwrap().len() == 16);
}

#[test]
fn launcher_json_stays_current_until_the_catalogue_changes() {
    // py: "a launcher.json of the current catalogue is not stale", "one written by an older app is, so its next boot writes the new rows", "and so is a game with none"
    let (fx, mut doc, out) = launcher_fixture();
    let tid = "0100B8E016F76000";
    assert!(!launcher_stale(&fx.ctx, tid));
    doc["catalogue"] = serde_json::json!("older-app");
    fs::write(&out, doc.to_string()).unwrap();
    assert!(launcher_stale(&fx.ctx, tid));
    assert!(launcher_stale(&fx.ctx, "0100000000000000"));
}

#[test]
fn a_launcher_row_the_player_set_reaches_the_boot_as_its_switch() {
    // py: "a launcher row the player set reaches the boot as its switch", "a switch the game's launcher does not offer does not", "Low latency on sets the drain and the late latch it needs"
    let (fx, _doc, _out) = launcher_fixture();
    let tid = "0100B8E016F76000";
    write(
        fx.p(&format!("games/{tid}/settings.tsv")),
        "DM_DYNAMIC_RESOLUTION\t1\nDM_LANG\tfr\nDM_FULLSCREEN\t1\nfps\t30\n",
    );
    let got = apply_game_settings(&fx.ctx);
    assert_eq!(
        got.get("DM_DYNAMIC_RESOLUTION").map(String::as_str),
        Some("1")
    );
    assert_eq!(got.get("DM_LANG").map(String::as_str), Some("fr"));
    assert_eq!(got.get("DM_FPS_TARGET").map(String::as_str), Some("30"));
    assert!(!got.contains_key("DM_FULLSCREEN"), "{got:?}");
    for k in got.keys() {
        fx.env.remove(k);
    }
    write(
        fx.p(&format!("games/{tid}/settings.tsv")),
        "DM_PRESENT_DRAIN\t1\n",
    );
    let got = apply_game_settings(&fx.ctx);
    assert_eq!(got.get("DM_PRESENT_DRAIN").map(String::as_str), Some("1"));
    assert_eq!(
        got.get("DM_PRESENT_LATCH").map(String::as_str),
        Some("1"),
        "the drain needs the late latch"
    );
}

#[test]
fn the_games_graphics_api() {
    // py: "the game gfx API: vulkan without nvnBootstrapLoader, nvn when unreadable"
    let fx = Fx::new();
    write(fx.p("vk.bin"), b"a\0vkCreateInstance\0b");
    assert_eq!(game_gfxapi(&fx.p("vk.bin")), "vulkan");
    assert_eq!(game_gfxapi(&fx.p("missing")), "nvn");
    write(
        fx.p("both.bin"),
        b"vkCreateInstance\0 .. nvnBootstrapLoader\0",
    );
    assert_eq!(
        game_gfxapi(&fx.p("both.bin")),
        "nvn",
        "nvn wins wherever it is in the file"
    );
    write(fx.p("gl.bin"), b"eglCreateWindowSurface\0 glShaderSource");
    assert_eq!(game_gfxapi(&fx.p("gl.bin")), "gl");
    // a needle that straddles a read block
    let mut big = vec![0u8; (4 << 20) - 5];
    big.extend_from_slice(b"nvnBootstrapLoader\0");
    big.extend_from_slice(&[0u8; 100]);
    write(fx.p("big.bin"), &big);
    assert_eq!(game_gfxapi(&fx.p("big.bin")), "nvn");
}

#[test]
fn launcher_json_is_written_like_python_json_dump() {
    let v = serde_json::json!({"a": [], "b": {}, "n": "caf\u{e9} \u{1F600}", "z": [1, {"k": "v"}]});
    assert_eq!(
        to_python_json(&v),
        "{\n \"a\": [],\n \"b\": {},\n \"n\": \"caf\\u00e9 \\ud83d\\ude00\",\n \"z\": [\n  1,\n  {\n   \"k\": \"v\"\n  }\n ]\n}"
    );
}

// ---- the address space against the game's block ------------------------------------------------------------------------------------------------------------------------

fn space(free: u64, largest: u64, extended: bool) -> Option<AddressSpace> {
    Some(AddressSpace {
        free,
        largest,
        extended,
        increased_memory: -1,
    })
}

#[test]
fn a_12_5_gb_block_on_a_standard_address_space_needs_the_extended_build() {
    // py: "a 12.5 GB block on a standard address space is said to need the extended build", "and fits with extended addressing, with no warning", "a 5 GB block in a 6 GB hole: no warning"
    let mut fx = Fx::new();
    fx.profile("title_id\t0400486024DEC000\nmem_reserve\t0x320000000\n");
    fx.mem.set_address_space(space(4 * G, 3 * G, false));
    let got = check_address_space(&fx.ctx).unwrap();
    assert_eq!(got.reserve, 0x3_2000_0000);
    assert!(fx.log.contains("extended-addressing build"));
    fx.mem.set_address_space(space(50 * G, 40 * G, true));
    fx.log.clear();
    check_address_space(&fx.ctx);
    assert!(
        fx.log.contains("EXTENDED") && !fx.log.contains("[RED]"),
        "{:?}",
        fx.log.lines()
    );
    fx.profile("title_id\t0100DFA027504000\nmem_reserve\t0x140000000\n");
    fx.mem.set_address_space(space(4 * G, 6 * G, false));
    fx.log.clear();
    check_address_space(&fx.ctx);
    assert!(!fx.log.contains("[RED]"));
    // not on a device
    fx.mem.set_address_space(None);
    assert!(check_address_space(&fx.ctx).is_none());
}

// ---- each game keeps its icon ------------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn each_game_keeps_its_icon() {
    // py: "the extraction's icon is kept under the game's own folder", "and clears an earlier "no icon" marker"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("extract_out", &fx.p("extract").to_string_lossy());
    write(fx.p("extract/icon.jpg"), "JPEG");
    write(fx.p(&format!("games/{TID}/icon.none")), "no keys then");
    let got = save_game_icon(&fx.ctx).unwrap();
    assert_eq!(fs::read_to_string(got).unwrap(), "JPEG");
    assert!(!fx.p(&format!("games/{TID}/icon.none")).exists());
}

// ---- the heap region sized before a game's first boot (trap 46) ---------------------------------------------------------------------------------------------------------------------

fn presize_case(
    largest: u64,
    survey_note: &str,
    guest_log: Option<&str>,
) -> (
    std::collections::BTreeMap<String, i128>,
    String,
    std::collections::BTreeMap<String, i128>,
    Arc<MemLog>,
) {
    let mut fx = Fx::new();
    fx.mem.set_address_space(space(largest, largest, false));
    write(
        fx.p(&format!("profile-derived/{SP}/game.tsv")),
        format!(
            "title_id\t{SP}\nmodules\tmain sdk\n# mem_reserve 0x140000000: {survey_note}\nmem_reserve\t0x140000000\nmem_alias\t0x118000000\nmem_stack\t0x10000000\nsys_resource\t0x1000000\n"
        ),
    );
    fx.profile(&format!("title_id\t{SP}\n"));
    if let Some(g) = guest_log {
        write(fx.p("gameapp.guest.log"), g.replace("%s", SP));
    }
    fx.ctx.adapters.profile = Arc::new(RecordingProfile::default());
    let got = presize_heap(&fx.ctx);
    let gt = fs::read_to_string(fx.p(&format!("profile-derived/{SP}/game.tsv"))).unwrap();
    let again = presize_heap(&fx.ctx);
    (got, gt, again, fx.log.clone())
}

#[test]
fn the_heap_region_is_sized_before_a_games_first_boot() {
    // py: the five presize checks
    let (got, gt, again, lines) = presize_case(64 * G, "main sets a heap", None);
    assert_eq!(
        got.get("mem_reserve"),
        Some(&0x2_0800_0000),
        "{got:?} {:?}",
        lines.lines()
    );
    assert!(gt.contains("mem_alias\t0x118000000") && gt.contains("mem_reserve\t0x208000000"));
    assert!(again.is_empty(), "once only");
    let (got, gt, _, _) = presize_case(4 * G, "main sets a heap", None);
    assert_eq!(
        got.get("mem_alias"),
        Some(&0x5000_0000),
        "little room, a game that sets a heap: the heap region comes out of the alias region"
    );
    assert!(gt.contains("mem_reserve\t0x140000000"));
    let (got, gt, _, _) = presize_case(4 * G, "no heap set (Unity)", None);
    assert!(
        got.is_empty() && gt.contains("mem_alias\t0x118000000"),
        "a game that sets no heap keeps its alias region"
    );
    let (got, ..) = presize_case(
        64 * G,
        "main sets a heap",
        Some("[dm] build: x, profile %s\n[dm] [mem] heap 3.14 GB\n"),
    );
    assert!(
        got.is_empty(),
        "a game already booted on this device is left to the heap check"
    );
}

// ---- the alias region from the address space (trap 47) ----------------------------------------------------------------------------------------------------------------------

#[test]
fn the_alias_region_from_the_address_space() {
    // py: "64 GB in one hole: a 48 GB alias region", "and the block check counts it (48.625 GB), with no warning in 64 GB", "a small address space keeps the profile's alias region", "a block taken at launch (DM_AS_AT) is never resized"
    let mut fx = Fx::new();
    fx.profile("title_id\t01008F2005154000\nmem_reserve\t0x140000000\nmem_alias\t0x118000000\n");
    fx.mem.set_address_space(space(64 * G, 64 * G, false));
    assert_eq!(set_alias_size(&fx.ctx), Some(48 * G as i128));
    assert_eq!(fx.env.get("DM_ALIAS_SIZE"), Some(format!("{:#x}", 48 * G)));
    fx.log.clear();
    check_address_space(&fx.ctx);
    assert!(
        fx.log
            .any(|l| l.contains("48.62 GB") || l.contains("48.63 GB"))
            && !fx.log.contains("[RED]"),
        "{:?}",
        fx.log.lines()
    );
    fx.env.remove("DM_ALIAS_SIZE");
    fx.mem.set_address_space(space(4 * G, 3 * G, false));
    assert!(set_alias_size(&fx.ctx).is_none() && !fx.env.truthy("DM_ALIAS_SIZE"));
    fx.env.set("DM_AS_AT", "0x7000000000");
    fx.mem.set_address_space(space(64 * G, 64 * G, false));
    assert!(set_alias_size(&fx.ctx).is_none() && !fx.env.truthy("DM_ALIAS_SIZE"));
}

// ---- a library split by another version of the split is built again; the DLC set ------------------------------------------------------------------------------------------------------

#[test]
fn a_library_split_by_another_version_of_the_split_is_stale() {
    // py: "a library made here before the split field is out of date", "one made by this split is not", "an imported pack is never second-guessed"
    let fx = Fx::new();
    let pack = fx.p("shaders/01008F2005154000");
    write(
        pack.join("pack.txt"),
        "game\t01008F2005154000\nchain\tcc0f2129e1de\nmade\ton this device, 2026-11-18\nshaders\t623\n",
    );
    assert_eq!(stale_split(&pack, "abc123def456"), Some("none".to_string()));
    write(
        pack.join("pack.txt"),
        "made\ton this device, x\nsplit\tabc123def456\n",
    );
    assert_eq!(stale_split(&pack, "abc123def456"), None);
    assert_eq!(
        stale_split(&pack, "-"),
        None,
        "an unknown split version never makes a library stale"
    );
    write(
        pack.join("pack.txt"),
        "game\t0100DFA027504000\nchain\tcc0f2129e1de\n",
    );
    assert_eq!(stale_split(&pack, "abc123def456"), None);
}

#[test]
fn the_dlc_set_a_library_was_split_with() {
    // py: the DLC block
    let fx = Fx::new();
    let tid = "01008F2005154000";
    assert!(aoc_images(&fx.ctx, tid).is_empty() && dlc_signature(&[]) == "-");
    let aoc = fx.p(&format!("games/{tid}/aoc"));
    for (name, size) in [
        ("10.img", 3),
        ("2.img", 2),
        ("1.img", 1),
        (".1.img", 9),
        ("notes.txt", 4),
    ] {
        write(aoc.join(name), "x".repeat(size));
    }
    let imgs = aoc_images(&fx.ctx, tid);
    assert_eq!(
        imgs.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        ["1.img", "2.img", "10.img"],
        "2 before 10, hidden and other files left out"
    );
    let sig = dlc_signature(&imgs);
    assert_eq!(sig, "1:1,2:2,10:3");
    let pack = fx.p(&format!("shaders/{tid}"));
    write(
        pack.join("pack.txt"),
        "made\ton this device, x\nsplit\tabc\n",
    );
    assert_eq!(
        stale_dlc(&pack, &sig),
        Some(("-".into(), sig.clone())),
        "a library made here before the dlc field counts as split with none"
    );
    assert_eq!(stale_dlc(&pack, "-"), None);
    write(
        pack.join("pack.txt"),
        format!("made\ton this device, x\ndlc\t{sig}\n"),
    );
    assert_eq!(stale_dlc(&pack, &sig), None);
    write(aoc.join("2.img"), "a newer version");
    assert!(stale_dlc(&pack, &dlc_signature(&aoc_images(&fx.ctx, tid))).is_some());
    write(
        pack.join("pack.txt"),
        format!("game\t{tid}\nchain\tcc0f2129e1de\n"),
    );
    assert_eq!(
        stale_dlc(&pack, &sig),
        None,
        "an imported pack is never second-guessed for its DLC"
    );
}

// ---- stage 3 under the memory ceiling ------------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn threads_for_the_memory_left() {
    // py: the shader_jobs_for_memory and rewrite_jobs checks (the GIL ones do not exist in Rust: threads always run in parallel)
    let log = MemLog::new();
    assert_eq!(
        shader_jobs_for_memory(
            6,
            &FakeMem::new(3218 * MIB),
            &*log,
            SHADER_PER_JOB,
            "translation"
        ),
        4,
        "3.1 GB left after the rewrite: 4 threads, not 6"
    );
    assert!(log.contains("4 translation threads instead of 6"));
    assert_eq!(
        shader_jobs_for_memory(
            6,
            &FakeMem::new((1.2 * G as f64) as u64),
            &NullLog,
            SHADER_PER_JOB,
            "translation"
        ),
        1
    );
    assert_eq!(
        shader_jobs_for_memory(
            6,
            &FakeMem::new(6 * G),
            &NullLog,
            SHADER_PER_JOB,
            "translation"
        ),
        6
    );
    assert_eq!(
        shader_jobs_for_memory(6, &FakeMem::new(0), &NullLog, SHADER_PER_JOB, "translation"),
        6,
        "no ceiling to read (simulator): unchanged"
    );
    let cfg = |s: &str| ibuilder_pipeline::config::Config::parse(s);
    assert_eq!(
        rewrite_jobs(&cfg(""), None, &FakeMem::new(5500 * MIB), 6, (0, false)),
        4,
        "the rewrite of main: 4 threads, 6 cores, cool, FlareStore memory"
    );
    assert_eq!(
        rewrite_jobs(&cfg(""), None, &FakeMem::new(0), 6, (0, true)),
        2,
        "2 in Low performance mode"
    );
    assert_eq!(
        rewrite_jobs(&cfg(""), None, &FakeMem::new(0), 6, (2, false)),
        1,
        "1 when the iPhone is hot"
    );
    assert_eq!(
        rewrite_jobs(&cfg(""), None, &FakeMem::new(2800 * MIB), 6, (0, false)),
        1,
        "1 under a SideStore ceiling (the serial rewrite alone peaks near 3 GiB)"
    );
    assert_eq!(
        rewrite_jobs(&cfg("rewrite_jobs=3"), None, &FakeMem::new(0), 6, (2, true)),
        3,
        "the config overrides it"
    );
    assert_eq!(
        shader_jobs_for_memory(6, &FakeMem::new(5283 * MIB), &NullLog, 900 * MIB, "reading"),
        4,
        "reading South Park with 5.3 GB left: 4 reading threads"
    );
}

#[test]
fn the_memory_gate_holds_a_second_translation_back_when_little_is_left() {
    // py: "the gate: with little left, a second translation waits for the first to end", "with room, they run together"
    use ibuilder_pipeline::gate::ShaderMemoryGate;
    let order = Arc::new(Mutex::new(Vec::<(String, u32)>::new()));
    let run_two = |mem: Arc<FakeMem>| {
        let gate = Arc::new(ShaderMemoryGate::new(
            mem,
            Arc::new(NullLog),
            None,
            RealClock::new(),
        ));
        order.lock().unwrap().clear();
        let mut hs = vec![];
        for i in 1..=2u32 {
            let (g, o) = (gate.clone(), order.clone());
            hs.push(std::thread::spawn(move || {
                g.run(|| {
                    o.lock().unwrap().push(("start".into(), i));
                    std::thread::sleep(std::time::Duration::from_millis(400));
                    o.lock().unwrap().push(("end".into(), i));
                })
            }));
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        for h in hs {
            h.join().unwrap();
        }
        order.lock().unwrap().clone()
    };
    let s = |a: &str, i: u32| (a.to_string(), i);
    assert_eq!(
        run_two(Arc::new(FakeMem::new(500 * MIB))),
        vec![s("start", 1), s("end", 1), s("start", 2), s("end", 2)]
    );
    let o = run_two(Arc::new(FakeMem::new(4 * G)));
    assert_eq!(
        &o[..2],
        &[s("start", 1), s("start", 2)],
        "with room, they run together: {o:?}"
    );
}

// ---- free-form engine switches from env.txt ----------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn env_txt_the_games_file_wins_and_only_dm_and_mtl_names_are_taken() {
    // py: "env.txt: the game's file wins over the global one, DM_/MTL_ only", "a switch already set (the settings sheet) is left alone"
    let mut fx = Fx::new();
    fx.profile(&format!("title_id\t{TID}\n"));
    write(
        fx.p("env.txt"),
        "# global\nDM_PRESENT_DRAIN=1\nDM_NVN_BARRIER=full\nHOME=/evil\n",
    );
    write(
        fx.p(&format!("games/{TID}/env.txt")),
        "DM_NVN_BARRIER=game   # this game\nlower=1\nMTL_HUD_ENABLED=1\n",
    );
    fx.env.set("DM_PRESENT_DRAIN", "0"); // set by the settings sheet: wins
    let got = apply_env_files(&fx.ctx);
    let want: std::collections::BTreeMap<String, String> =
        [("DM_NVN_BARRIER", "game"), ("MTL_HUD_ENABLED", "1")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
    assert_eq!(got, want);
    assert_eq!(fx.env.get("HOME"), None);
    assert_eq!(fx.env.get("DM_PRESENT_DRAIN").as_deref(), Some("0"));
    assert!(fx.log.contains("already set to '0'"));
}

// ---- "Parallel shader preparation" ------------------------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn parallel_shader_preparation_is_on_unless_the_player_turned_it_off() {
    // py: the parallel_shaders block (the free-threaded / GIL combinations collapse: Rust has no GIL)
    let fx = Fx::new();
    assert!(shader_parallelism(&fx.ctx).0, "no settings file is ON");
    write(
        fx.p("gameapp.settings.tsv"),
        "other\t1\nparallel_shaders\t0\n",
    );
    assert!(!shader_parallelism(&fx.ctx).0);
    write(
        fx.p("gameapp.settings.tsv"),
        "other\t0\nparallel_shaders\t1\n",
    );
    assert!(shader_parallelism(&fx.ctx).0);
}

// ---- the shared shader translation cache (trap 51) ----------------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn the_shared_shader_cache() {
    // py: "the shader cache is Documents/shadercache, shared by every game", "another chain's files are removed", "over its limit it starts again", "shader_cache=0 keeps the old behaviour", "its default limit is 2 GB"
    // The Rust build's cache is the FOLDER cas-<chain>.dmsb (one file per shader under <hh>/), not the engine's SQLite file: what it keeps, what it removes (files AND folders), how the limit counts it.
    let mut fx = Fx::new();
    let c = fx.p("shadercache");
    let mine = "cas-b24b25240cad.dmsb";
    write(c.join("cas-19d183a2b462.db"), "old chain");
    write(c.join("cas-19d183a2b462.db-wal"), "old wal");
    write(
        c.join("cas-19d183a2b462.dmsb/ab/entry1"),
        "old chain, a folder",
    );
    write(
        c.join("cas-19d183a2b462.dmsb/cd/entry2"),
        "old chain, another sub-folder",
    );
    write(
        c.join("cas-b24b25240cad.db"),
        "the engine's SQLite store of this very chain: unreadable here",
    );
    write(c.join("cas-b24b25240cad.db-shm"), "its shm");
    write(
        c.join(format!("{mine}/ab/{}", "ab".repeat(32))),
        "x".repeat(1000),
    );
    write(
        c.join("unrelated.txt"),
        "not ours: it does not start with cas",
    );
    assert_eq!(
        shader_cache_folder(&fx.ctx, "b24b25240cad", false),
        Some(c.clone())
    );
    assert_eq!(
        ibuilder_pipeline::util::list_names(&c),
        vec![mine, "unrelated.txt"],
        "every other cas* (files AND folders: remove_file cannot remove a folder), the same chain's SQLite file included, is removed; this chain's folder and what is not a cache are kept"
    );
    assert!(
        c.join(format!("{mine}/ab/{}", "ab".repeat(32))).is_file(),
        "the entries of this chain's cache are untouched"
    );
    assert!(
        fx.log
            .contains("cas-19d183a2b462.dmsb removed (another chain version")
            && fx
                .log
                .contains("cas-b24b25240cad.db removed (the engine's SQLite cache"),
        "{:?}",
        fx.log.lines()
    );
    assert!(
        !fx.log.contains("could not be removed"),
        "{:?}",
        fx.log.lines()
    );
    assert_eq!(
        ibuilder_pipeline::shaderpack::shader_cache_name("b24b25240cad"),
        mine
    );
    // the limit counts the folder's entries (they are in sub-folders): over it, the cache starts again
    fx.set_cfg("shader_cache_gib", "0.0000001");
    fx.log.clear();
    shader_cache_folder(&fx.ctx, "b24b25240cad", false);
    assert!(
        ibuilder_pipeline::util::list_names(&c).is_empty() && fx.log.contains("started again"),
        "over its limit it starts again"
    );
    fx.set_cfg("shader_cache_gib", "0.0000001");
    write(
        c.join(format!("{mine}/ab/{}", "ab".repeat(32))),
        "x".repeat(1000),
    );
    fx.log.clear();
    shader_cache_folder(&fx.ctx, "b24b25240cad", true);
    assert!(
        c.join(mine).join("ab").is_dir() && fx.log.contains("kept for now"),
        "resuming: not started again over the limit"
    );
    fx.ctx.cfg.remove("shader_cache_gib");
    fx.set_cfg("shader_cache", "0");
    assert_eq!(
        shader_cache_folder(&fx.ctx, "x", false),
        None,
        "shader_cache=0 keeps the old behaviour"
    );
    fx.ctx.cfg.remove("shader_cache");
    // the default limit is 2 GB: a 1 GB cache is within it
    fx.log.clear();
    shader_cache_folder(&fx.ctx, "b24b25240cad", false);
    assert!(fx.log.contains("limit 2.00 GB"), "{:?}", fx.log.lines());
    assert!(
        c.join(mine).join("ab").is_dir(),
        "within the limit nothing goes"
    );
}

#[test]
fn a_symlink_named_like_a_stale_cache_is_removed_and_never_followed() {
    let fx = Fx::new();
    let c = fx.p("shadercache");
    write(c.join("cas-b24b25240cad.dmsb/00/e"), "mine");
    // a dangling symlink (and one to a folder that must survive) named like a stale cache is removed as a file: `remove_dir_all` on it would follow it, `remove_file` on a real folder fails
    write(
        fx.p("elsewhere/keep"),
        "outside the cache folder: must survive",
    );
    std::os::unix::fs::symlink(fx.p("elsewhere"), c.join("cas-0001.dmsb")).unwrap();
    std::os::unix::fs::symlink(fx.p("nowhere"), c.join("cas-0000.dmsb")).unwrap();
    assert_eq!(
        shader_cache_folder(&fx.ctx, "b24b25240cad", false),
        Some(c.clone())
    );
    assert_eq!(
        ibuilder_pipeline::util::list_names(&c),
        vec!["cas-b24b25240cad.dmsb"]
    );
    assert!(
        fx.p("elsewhere/keep").is_file(),
        "the link's target is not touched"
    );
}

// ---- a distribution certificate: a dlopen from Documents is refused, Play becomes Export ---------------------------------------------------------------------------------------------------------

#[test]
fn a_sandbox_refused_dlopen_is_recorded_as_a_load_this_install_cannot_do() {
    // py: "a sandbox-refused dlopen is recorded", "and its boot request is withdrawn", "another dlopen failure is not that", "a boot that ran clears the record"
    let fx = Fx::new();
    write(fx.p("gameapp.boot-only"), "attempted pid 1");
    write(
        fx.p("gameapp.guest.log"),
        "[gameapp] dlopen FAILED: dlopen(/x/DmGameProbe, 0x0006): tried: '/x' (file system sandbox blocked mmap() of '/x')\n",
    );
    assert!(note_boot_result(&fx.ctx, 3));
    assert!(fx.p("gameapp.load-blocked").is_file() && fx.log.contains("Export as .ipa"));
    assert!(
        !fx.p("gameapp.boot-only").exists(),
        "no 'starts again' loop"
    );
    write(
        fx.p("gameapp.guest.log"),
        "[gameapp] dlopen FAILED: image not found\n",
    );
    write(fx.p("gameapp.boot-only"), "attempted pid 2");
    assert!(!note_boot_result(&fx.ctx, 3));
    assert!(fx.p("gameapp.boot-only").exists());
    assert!(!note_boot_result(&fx.ctx, 0));
    assert!(
        !fx.p("gameapp.load-blocked").exists(),
        "an install with a development certificate"
    );
}

/// Review r5 P1: the guest log is opened `O_APPEND` and keeps every boot: the refusal is looked for in the LAST boot's section (from its banner on), however far from the end it is (a 16 KiB
/// tail missed one followed by more output), and an earlier boot's refusal is not this boot's.
#[test]
fn the_sandbox_refusal_is_found_anywhere_in_the_last_boot_and_only_there() {
    let fx = Fx::new();
    let banner = |pid: u32| {
        format!("\n{GUEST_RUN_BANNER}{pid} (what the host and the game wrote to fd 1/2) ====\n")
    };
    let filler = |n: usize| {
        (0..n)
            .map(|i| format!("[dm] guest line {i:05} {}\n", "x".repeat(60)))
            .collect::<String>()
    };
    // three earlier boots that ran (tens of KB), then this one: the refusal, then 40 KB of other output after it
    let mut log = String::new();
    for pid in 1..=3 {
        log.push_str(&banner(pid));
        log.push_str(&filler(300));
    }
    log.push_str(&banner(4));
    log.push_str("[gameapp] dlopen FAILED: dlopen(/x/DmGameProbe, 0x0006): tried: '/x' (file system sandbox blocked mmap() of '/x')\n");
    log.push_str(&filler(500));
    let refusal_from_end = log.len() - log.find("sandbox blocked mmap").unwrap();
    assert!(
        refusal_from_end > 16 << 10 && log.len() > 64 << 10,
        "{refusal_from_end} of {}",
        log.len()
    );
    write(fx.p("gameapp.guest.log"), &log);
    write(fx.p("gameapp.boot-only"), "attempted pid 4");
    assert!(
        note_boot_result(&fx.ctx, 3),
        "a refusal {refusal_from_end} bytes from the end"
    );
    assert!(fx.p("gameapp.load-blocked").is_file() && !fx.p("gameapp.boot-only").exists());

    // a later boot that failed for another reason: the earlier refusal (now in an older section) is not this one's
    fs::remove_file(fx.p("gameapp.load-blocked")).unwrap();
    log.push_str(&banner(5));
    log.push_str("[gameapp] dlopen FAILED: image not found\n");
    write(fx.p("gameapp.guest.log"), &log);
    write(fx.p("gameapp.boot-only"), "attempted pid 5");
    assert!(!note_boot_result(&fx.ctx, 3));
    assert!(!fx.p("gameapp.load-blocked").exists() && fx.p("gameapp.boot-only").exists());
    assert!(!last_guest_run_refused(&fx.p("no-such.log")));
}

// ---- an exported game: its shader library is in its bundle ---------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn an_exported_game_takes_its_shader_library_from_its_bundle() {
    // py: "an exported game takes its shader library from its bundle (shader_packs=)", "and builds nothing there"
    let mut fx = Fx::new();
    let packs = fx.p("Bundle.app/ExportedGame/shaders");
    let tid = "0100B8E016F76000";
    write(packs.join(format!("{tid}/shader_map.tsv")), rows(3));
    write(packs.join(format!("{tid}/b-b24b25240cad.img")), "lib");
    fx.profile(&format!("title_id\t{tid}\n"));
    fx.set_cfg("shader_packs", &packs.to_string_lossy());
    use_config_paths(&mut fx.ctx);
    assert!(
        fx.log
            .contains("an exported game: its library is in its bundle")
    );
    fx.set_cfg("shader_chain", "b24b25240cad");
    let got = set_shader_pack(&fx.ctx);
    assert_eq!(got.as_deref(), Some(packs.join(tid).as_path()));
    assert_eq!(
        fx.env.get("DM_MSL_DIR"),
        Some(packs.join(tid).to_string_lossy().into_owned())
    );
    fx.set_cfg("shader_build", "0");
    assert_eq!(
        build_shaders_on_device(&mut fx.ctx).as_deref(),
        Some(packs.join(tid).as_path()),
        "the library of its chain is found: nothing to build"
    );
}

// ---- space: the extracted code goes after a build, flat/ is hard links ---------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn the_extracted_code_goes_after_a_build() {
    // py: "the extracted code is kept when the harness asks", "and removed otherwise, saying how much it freed", "a missing folder is nothing to do"
    let mut fx = Fx::new();
    let exe = fx.p("extract/ExeFS");
    write(exe.join("main"), "m".repeat(4096));
    fx.set_cfg("exe_dir", &exe.to_string_lossy());
    fx.set_cfg("keep_exefs", "1");
    assert!(drop_exefs(&fx.ctx) == 0 && exe.is_dir());
    fx.ctx.cfg.remove("keep_exefs");
    assert_eq!(drop_exefs(&fx.ctx), 4096);
    assert!(!exe.exists() && fx.log.contains("ExeFS removed"));
    assert_eq!(drop_exefs(&fx.ctx), 0);
}

#[test]
fn flat_is_made_of_hard_links_to_the_rewrite_not_a_second_copy() {
    // py: "flat/ is made of hard links to the rewrite, not a second copy" (the Python test grepped the source; this one checks the inodes)
    use std::os::unix::fs::MetadataExt;
    let fx = Fx::new();
    let patched = fx.p("rewrite");
    for m in ["main", "sdk"] {
        for s in [".patched.bin", ".fixups.tsv", ".gates.tsv"] {
            write(patched.join(m).join(format!("{m}{s}")), format!("{m}{s}"));
        }
    }
    let n = ibuilder_pipeline::assemble::make_flat(
        &patched,
        &fx.p("flat"),
        &["main".to_string(), "sdk".to_string()],
    )
    .unwrap();
    assert_eq!(n, 6);
    assert_eq!(
        fs::metadata(patched.join("main/main.patched.bin"))
            .unwrap()
            .ino(),
        fs::metadata(fx.p("flat/main.patched.bin")).unwrap().ino()
    );
    let err = ibuilder_pipeline::assemble::make_flat(
        &patched,
        &fx.p("flat2"),
        &["main".into(), "subsdk9".into()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("the rewrite wrote no subsdk9"));
}

// ---- topping up a pack that is already on the phone (docs/CONFORMITY-AUDIT.md, device log 2026-11-19: "pipeline prewarm: seed list does not exist") ------------

fn installed_pack(fx: &Fx) -> PathBuf {
    let pack = fx.p(&format!("shaders/{TID}"));
    write(pack.join("shader_map.tsv"), rows(3));
    write(
        pack.join("c1_full.tsv"),
        "# chain b-19d183a2b462 inputs 8ea0535d1d1c\n",
    );
    write(pack.join("b-19d183a2b462.img"), "IMG-LIBRARY-BYTES");
    write(
        pack.join("pack.txt"),
        format!("game\t{TID}\nchain\t19d183a2b462\nmade\ton this device, 2026-11-19\nshaders\t3\n"),
    );
    pack
}

fn bundle_game(fx: &Fx, files: &[(&str, &str)]) {
    for (name, text) in files {
        write(fx.p(&format!("dm/games/{TID}/{name}")), text);
    }
}

fn names_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn an_existing_pack_gets_the_registered_games_missing_tables_and_nothing_else() {
    // the bug: Mario Kart's pack was built when the bundle held only game.tsv; the host then said "[RED] pipeline prewarm: seed list does not exist: Documents/shaders/<id>/pso_manifest.tsv"
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("shader_chain", "19d183a2b462");
    let pack = installed_pack(&fx);
    bundle_game(
        &fx,
        &[
            ("game.tsv", "title_id\tx\n"),
            ("pso_manifest.tsv", "S\tk_b\tvertex\nR\t00aa\n"),
            ("bf_want.tsv", "# generated\n"),
            ("bench.tsv", "scene\t1\n"),
            ("README.md", "readme"),
            ("shader_map.tsv", "THE BUNDLE'S OWN MAP: never copied\n"),
            ("shader_pairs.tsv", "split table: never copied\n"),
            ("c1_full.tsv", "split table: never copied\n"),
        ],
    );
    let library = fs::read(pack.join("b-19d183a2b462.img")).unwrap();
    let map = fs::read(pack.join("shader_map.tsv")).unwrap();
    let c1 = fs::read(pack.join("c1_full.tsv")).unwrap();

    let got = set_shader_pack(&fx.ctx);
    assert_eq!(got.as_deref(), Some(pack.as_path()));
    assert_eq!(
        fs::read_to_string(pack.join("pso_manifest.tsv")).unwrap(),
        "S\tk_b\tvertex\nR\t00aa\n",
        "the seed list the host reads beside the shader map"
    );
    assert!(pack.join("bf_want.tsv").is_file() && pack.join("bench.tsv").is_file());
    for never in [
        "game.tsv",
        "README.md",
        "shader_pairs.tsv",
        "dual_want.tsv",
        "geo_mesh.tsv",
    ] {
        assert!(!pack.join(never).exists(), "{never} must not be added");
    }
    assert_eq!(
        fs::read(pack.join("shader_map.tsv")).unwrap(),
        map,
        "the map is never touched"
    );
    assert_eq!(
        fs::read(pack.join("c1_full.tsv")).unwrap(),
        c1,
        "the split's own tables are never replaced"
    );
    assert_eq!(
        fs::read(pack.join("b-19d183a2b462.img")).unwrap(),
        library,
        "the library is never touched"
    );
    assert!(
        names_in(&pack).iter().all(|n| !n.ends_with(".part")),
        "no half-written file is left: {:?}",
        names_in(&pack)
    );
    assert!(
        fx.log.any(|l| l.contains("topped up")
            && l.contains("pso_manifest.tsv")
            && l.contains("bf_want.tsv")
            && l.contains("bench.tsv")),
        "{:?}",
        fx.log.lines()
    );
    // the usual pack lines are still there, and DM_NVN_SHMAP still names the pack's map
    assert_eq!(
        fx.env.get("DM_NVN_SHMAP"),
        Some(pack.join("shader_map.tsv").to_string_lossy().into_owned())
    );
}

#[test]
fn a_top_up_never_overwrites_a_file_the_pack_has_and_the_second_boot_adds_nothing_and_says_nothing()
{
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    let pack = installed_pack(&fx);
    write(
        pack.join("pso_manifest.tsv"),
        "the pack's own seed list (a Mac pack, or the build's copy)\n",
    );
    write(pack.join("dual_want.tsv"), ""); // even an empty one is the pack's
    bundle_game(
        &fx,
        &[
            ("pso_manifest.tsv", "BUNDLE\n"),
            ("dual_want.tsv", "BUNDLE\n"),
            ("bf_want.tsv", "# generated\n"),
        ],
    );
    assert!(top_up_pack(&fx.ctx, TID, &pack) == vec!["bf_want.tsv".to_string()]);
    assert_eq!(
        fs::read_to_string(pack.join("pso_manifest.tsv")).unwrap(),
        "the pack's own seed list (a Mac pack, or the build's copy)\n"
    );
    assert_eq!(fs::read_to_string(pack.join("dual_want.tsv")).unwrap(), "");
    let lines_before = fx.log.lines().len();
    assert!(top_up_pack(&fx.ctx, TID, &pack).is_empty(), "idempotent");
    assert!(set_shader_pack(&fx.ctx).is_some());
    assert!(
        !fx.log.lines()[lines_before..]
            .iter()
            .any(|l| l.contains("topped up")),
        "a boot with nothing to add logs nothing: {:?}",
        fx.log.lines()
    );
}

#[test]
fn the_derived_folders_table_wins_and_a_table_it_lacks_comes_from_the_bundle() {
    // after the heap check the derived folder holds game.tsv only (it copies just that file); a table the player edited there wins, as in the build
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    let pack = installed_pack(&fx);
    write(
        fx.p(&format!("profile-derived/{TID}/game.tsv")),
        "title_id\tx\n",
    );
    write(
        fx.p(&format!("profile-derived/{TID}/pso_manifest.tsv")),
        "DERIVED\n",
    );
    bundle_game(
        &fx,
        &[
            ("pso_manifest.tsv", "BUNDLE\n"),
            ("bf_want.tsv", "BUNDLE-BF\n"),
        ],
    );
    let got = top_up_pack(&fx.ctx, TID, &pack);
    assert_eq!(
        got,
        vec!["bf_want.tsv".to_string(), "pso_manifest.tsv".to_string()]
    );
    assert_eq!(
        fs::read_to_string(pack.join("pso_manifest.tsv")).unwrap(),
        "DERIVED\n"
    );
    assert_eq!(
        fs::read_to_string(pack.join("bf_want.tsv")).unwrap(),
        "BUNDLE-BF\n"
    );
}

#[test]
fn nothing_is_made_for_a_game_with_no_installed_pack_or_no_tables() {
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    bundle_game(&fx, &[("pso_manifest.tsv", "S\tk\tv\n")]);
    // no pack at all: the folder is not even created
    assert!(top_up_pack(&fx.ctx, TID, &fx.p(&format!("shaders/{TID}"))).is_empty());
    assert!(set_shader_pack(&fx.ctx).is_none());
    assert!(!fx.p(&format!("shaders/{TID}")).exists());
    // a folder with a header-only map is not an installed pack
    write(fx.p(&format!("shaders/{TID}/shader_map.tsv")), rows(0));
    assert!(top_up_pack(&fx.ctx, TID, &fx.p(&format!("shaders/{TID}"))).is_empty());
    assert!(!fx.p(&format!("shaders/{TID}/pso_manifest.tsv")).exists());
    // an installed pack, a bundle with no folder for the game: nothing to add, nothing said
    let pack = installed_pack(&fx);
    fs::remove_dir_all(fx.p(&format!("dm/games/{TID}"))).unwrap();
    assert!(top_up_pack(&fx.ctx, TID, &pack).is_empty());
    assert!(!fx.log.any(|l| l.contains("topped up")));
}

#[test]
fn an_exported_games_read_only_pack_in_its_bundle_is_left_alone() {
    // an exported game boots from <bundle>/ExportedGame/shaders (the config's shader_packs=): nothing is written into the bundle
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    let packs = fx.p("bundle/ExportedGame/shaders");
    let pack = packs.join(TID);
    write(pack.join("shader_map.tsv"), rows(3));
    write(pack.join("b-19d183a2b462.img"), "IMG");
    bundle_game(&fx, &[("pso_manifest.tsv", "S\tk\tv\n")]);
    fx.set_cfg("shader_packs", &packs.to_string_lossy());
    ibuilder_pipeline::shaderpack::use_config_paths(&mut fx.ctx);
    assert!(set_shader_pack(&fx.ctx).is_some());
    assert!(!pack.join("pso_manifest.tsv").exists());
    assert!(top_up_pack(&fx.ctx, TID, &pack).is_empty());
}

#[test]
fn a_build_input_added_to_an_existing_library_says_it_needs_a_new_build_and_a_comment_only_table_does_not()
 {
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    let pack = installed_pack(&fx);
    // the seed list alone (S/R rows, no mesh pair) and a comment-only bf_want: nothing the library lacks
    bundle_game(
        &fx,
        &[
            ("pso_manifest.tsv", "S\tk_b\tvertex\nR\t00aa\n"),
            ("bf_want.tsv", "# generated\n"),
        ],
    );
    top_up_pack(&fx.ctx, TID, &pack);
    assert!(
        !fx.log.any(|l| l.contains("[ORANGE]")),
        "{:?}",
        fx.log.lines()
    );
    // dual_want with a key, and a G row (a mesh pair) in a seed list: the library was built without them
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    let pack = installed_pack(&fx);
    bundle_game(
        &fx,
        &[
            ("pso_manifest.tsv", "S\tk_b\tvertex\nG\tv1\tg1\n"),
            ("dual_want.tsv", "0123456789abcdef_b\tcomment\n"),
            ("geo_mesh.tsv", "# none\n"),
        ],
    );
    top_up_pack(&fx.ctx, TID, &pack);
    assert!(
        fx.log.any(|l| l.contains("[ORANGE]")
            && l.contains("dual_want.tsv")
            && l.contains("pso_manifest.tsv")
            && !l.contains("geo_mesh.tsv")
            && l.contains("delete Documents/shaders/")),
        "{:?}",
        fx.log.lines()
    );
    assert!(
        pack.join("geo_mesh.tsv").is_file(),
        "added all the same: it is the game's table, comment-only or not"
    );
}

#[test]
fn stage_3_finding_the_library_already_there_tops_the_pack_up_too() {
    // a chain run (or a boot-only launch, which calls stage 3 first) on a game whose library is on disk: the pack is complete when it returns
    let mut fx = Fx::new();
    fx.ctx.bundle = fx.p("");
    fx.profile(&format!("title_id\t{TID}\n"));
    fx.set_cfg("shader_chain", "19d183a2b462");
    let pack = installed_pack(&fx);
    bundle_game(&fx, &[("pso_manifest.tsv", "S\tk\tv\n")]);
    assert_eq!(
        build_shaders_on_device(&mut fx.ctx).as_deref(),
        Some(pack.as_path())
    );
    assert!(pack.join("pso_manifest.tsv").is_file());
    assert!(
        fx.log
            .any(|l| l.contains("already holds a library of chain")),
        "{:?}",
        fx.log.lines()
    );
}

/// A settings catalogue with the engine's "Skip draws while they compile" row (default "1", the engine's own).
struct NowaitRows;
impl LauncherBase for NowaitRows {
    fn base_document(
        &self,
        _v: &ibuilder_pipeline::View<'_>,
        tid: &str,
        _m: &Path,
        _c: &Path,
        _g: &str,
        _d: Option<&Path>,
    ) -> ibuilder_pipeline::Result<serde_json::Value> {
        let mut d = settings_doc(tid);
        d["settings"] = serde_json::json!([{"env": "DM_PSO_NOWAIT", "kind": "toggle", "section": "Performance", "label": "Skip draws while they compile", "note": "n", "default": "1", "on": "1", "off": "0"}]);
        Ok(d)
    }
}

/// Review r4 B3: the sheet's "Skip draws while they compile" default is what the boot does for THIS game (`set_pso_nowait`: 0 for a seed list of at most 10,000 lines, the host's 1 above,
/// or the config's choice), not the engine's "1": the sheet showed ON while the game waited, and the player could not choose 1 (the app writes only a value that differs from the default).
#[test]
fn the_skip_draws_row_defaults_to_what_the_boot_will_do() {
    let mut fx = Fx::new();
    fx.ctx.adapters.launcher = Some(Arc::new(NowaitRows));
    let tid = "0100B8E016F76000";
    fx.profile(&format!("title_id\t{tid}\n"));
    fx.set_cfg("extract_out", &fx.p("extract").to_string_lossy());
    let main = fx.p("extract/rewrite/main");
    write(main.join("main.patched.bin"), b"x\0nvnBootstrapLoader\0y");
    write(main.join("main.fixups.tsv"), "");
    let row_default = |fx: &Fx| -> String {
        let out = write_launcher_settings(&fx.ctx, &main).expect("launcher.json");
        let doc: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out).unwrap()).unwrap();
        doc["settings"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["env"] == "DM_PSO_NOWAIT")
            .unwrap()["default"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let booted = |fx: &Fx| -> String {
        let probe = Fx::new();
        let mut ctx = ibuilder_pipeline::Ctx::builder(fx.docs())
            .env(probe.env.clone())
            .log(probe.log.clone())
            .mem(probe.mem.clone())
            .build();
        ctx.cfg = fx.ctx.cfg.clone();
        set_game_env(&ctx);
        probe
            .env
            .get("DM_PSO_NOWAIT")
            .unwrap_or_else(|| "1 (the host's)".into())
    };
    // no seed list at all, or a short one: the game waits, and the sheet says OFF
    assert_eq!(row_default(&fx), "0");
    assert_eq!(booted(&fx), "0");
    let seed =
        |n: usize| -> String { (0..n).map(|i| format!("R\t{i:016x}\n")).collect::<String>() };
    let pack = fx.p(&format!("shaders/{tid}"));
    write(pack.join("pso_manifest.tsv"), seed(1_516));
    assert!(!launcher_stale(&fx.ctx, tid));
    assert_eq!(row_default(&fx), "0");
    // a list over 10,000 lines: the host's policy, and the sheet says ON; a sheet written before the list came is stale
    write(
        pack.join("pso_manifest.tsv"),
        format!("# head\n{}", seed(10_001)),
    );
    assert!(
        launcher_stale(&fx.ctx, tid),
        "the default changed with the list: the next boot writes the sheet again"
    );
    assert_eq!(row_default(&fx), "1");
    assert!(!launcher_stale(&fx.ctx, tid));
    assert_eq!(booted(&fx), "1 (the host's)");
    // the list the pack will get (a registered game's, in the bundle) counts before the pack exists
    fs::remove_dir_all(&pack).unwrap();
    write(
        fx.ctx
            .bundle
            .join(format!("dm/games/{tid}/pso_manifest.tsv")),
        seed(25_163),
    );
    assert_eq!(row_default(&fx), "1");
    // the config's choice wins
    fx.set_cfg("pso_nowait", "0");
    assert_eq!(row_default(&fx), "0");
    fx.set_cfg("pso_nowait", "1");
    assert_eq!(row_default(&fx), "1");
}

// ---- a module the profile lists but the ExeFS does not carry (device log 2026-11-23, Mario Kart World base .nup: "the rewrite wrote no subsdk1") ------------------------

#[test]
fn a_module_missing_from_the_exefs_is_dropped_not_assembled() {
    let mut fx = Fx::new();
    fx.ctx.adapters.rewriter = Arc::new(CountingRewriter::default());
    let tid = "0100ABCD12340000";
    write(fx.p("exe/main"), "MAIN");
    write(fx.p("exe/sdk"), "SDK");
    fx.cfg("");
    fx.profile(&format!("title_id\t{tid}\n"));
    fx.set_cfg("exe_dir", &fx.p("exe").to_string_lossy());
    fx.set_cfg("modules", "main subsdk1 sdk");
    fx.set_cfg("extract_out", &fx.p("extract").to_string_lossy());
    let out = do_rewrite(&mut fx.ctx).unwrap();
    assert_eq!(fx.ctx.cfg.get("modules"), Some("main sdk"));
    assert!(fx.log.contains("dropped subsdk1"), "{:?}", fx.log.lines());
    assert!(out.join("main/main.patched.bin").is_file() && !out.join("subsdk1").exists());

    // without main there is no game: refused at stage 2, not after the shaders
    fs::remove_file(fx.p("exe/main")).unwrap();
    fx.set_cfg("modules", "main sdk");
    let err = do_rewrite(&mut fx.ctx).unwrap_err().to_string();
    assert!(err.contains("has no main"), "{err}");
}
