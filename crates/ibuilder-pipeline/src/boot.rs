//! Stage 5: the boot, and `boot_only`: stage 5 by itself, in a process that has run NO stage of the chain.
//!
//! Why this exists, measured (docs/DECISIONS.md trap 17): the chain's footprint at the boot is what it PEAKED at, not its live set -- releasing after every module collects real garbage
//! and the footprint does not move, because `malloc_zone_pressure_relief` hands back 0 MiB every time. So an in-process boot asks the guest to fit into what is left of the device's
//! 3365 MiB ceiling: `available 520 MiB` on the 2026-11-17 device run, and the app was killed inside the boot. A launch that has rewritten nothing has ~3.3 GiB free, which is the whole point
//! of running the boot there. The chain asks for this (`Documents/gameapp.boot-only`) whenever the margin it sees is smaller than `boot_floor_mib`.

use crate::adapters::BootRequest;
use crate::ctx::Ctx;
use crate::markers::{
    boot_only_consume, boot_only_mark_attempted, framework_is_export_only, note_boot_result,
    read_framework_stamp,
};
use crate::memory::{mem_line, release_memory};
use crate::mods::apply_mods;
use crate::profile::{
    apply_heap_check, check_address_space, presize_heap, profile_modules, set_alias_size,
    set_game_env,
};
use crate::settings::{
    apply_env_files, apply_game_settings, apply_preset, launcher_stale, write_launcher_settings,
};
use crate::shaderpack::{set_shader_map, set_shader_pack};
use crate::shaders::{LiveShaders, build_shaders};
use crate::stage::write_stage;
use crate::state::framework_matches;
use crate::util::{is_dir, is_file};
use std::fs;
use std::path::{Path, PathBuf};

pub const BOOT_STAGE_TEXT: &str =
    "booting the guest (dlopen + dm_host_probe_boot -> nnMain; quit from the gear menu)";

/// The steps both boots take before the guest runs, in their order: the heap region for a first boot, the heap check from the last boot's log, the player's settings, `env.txt`, the mods, the shader pack and
/// map, the save/temp folders, the alias region, the address-space check. Everything lands in `ctx.env` and is made visible to the host by `Env::commit` right before the boot.
pub fn prepare_boot(ctx: &Ctx) {
    presize_heap(ctx);
    apply_heap_check(ctx);
    apply_game_settings(ctx);
    apply_env_files(ctx);
    apply_preset(ctx); // the settings' Profile (Memory saver): its values for what the settings and env.txt left unset
    apply_mods(ctx); // the enabled mods' RomFS files and cheats: DM_ROMFS_OVERLAY, DM_CHEATS (a failure never stops the boot)
    set_shader_pack(ctx);
    set_shader_map(ctx);
    set_game_env(ctx);
    set_alias_size(ctx);
    check_address_space(ctx);
    set_bc_cpu(ctx); // last: a GPU without BC textures needs it whatever env.txt or a setting said
}

/// The one line a boot on a GPU without BC textures logs.
pub const BC_CPU_LINE: &str =
    "GPU without BC textures: textures decoded on the CPU (slower, more memory)";
/// The host's profile override (`hle_profile.c read_env`: `DM_PROF_<KEY>` on top of `DM_PROFILE`'s file, read at the first `dm_profile()`, after the dlopen): with `tex_bc_cpu` on,
/// `nvn_mtl.m mtl_init_once` sets `g_bc_cpu = !supportsBCTextureCompression && tex_bc_cpu` and decodes BC1-7 on the CPU (`nvn_bcdec.c`) instead of `MDIE("... has no BC texture formats")`.
pub const BC_CPU_ENV: &str = "DM_PROF_TEX_BC_CPU";

/// A GPU without BC textures (`Ctx::gpu_without_bc`, measured by the app through Metal): every game boots with `DM_PROF_TEX_BC_CPU=1`. A GPU with them (or one not measured) gets
/// nothing here: its environment and profile are exactly what they were. Returns whether it set the switch.
pub fn set_bc_cpu(ctx: &Ctx) -> bool {
    if !ctx.gpu_without_bc {
        return false;
    }
    if let Some(v) = ctx.env.get(BC_CPU_ENV).filter(|v| v.trim() != "1") {
        ctx.log(&format!("game env: {BC_CPU_ENV} was '{v}': set to 1, this GPU cannot read BC textures (the host would stop at the first one)"));
    }
    ctx.env.set(BC_CPU_ENV, "1");
    ctx.log(BC_CPU_LINE);
    true
}

/// What a boot returned: `(stage, guest)`, `(3, -1)` for no boot host.
pub fn run_boot(ctx: &mut Ctx, framework: &Path, flat: &Path, modules: &str) -> Option<(i32, i32)> {
    let host = ctx.adapters.boot.clone()?;
    let seconds = ctx.cfg.int_or("boot_seconds", 45);
    let profile = PathBuf::from(ctx.cfg.get_or("profile", ""));
    let romfs = ctx.cfg.get_or("romfs", "").to_string();
    let mut live = LiveShaders::start(ctx);
    ctx.env.commit(); // the host is C code in this process: it reads the real environment
    crate::diag::event(
        "boot_begin",
        serde_json::json!({"framework": framework.display().to_string(), "modules": modules, "romfs": romfs, "profile": profile.display().to_string(), "mem": crate::diag::mem_value(&ctx.mem.mem())}),
    );
    let r = host.boot(
        &BootRequest {
            framework,
            patched: flat,
            modules,
            profile: &profile,
            romfs: &romfs,
            seconds,
        },
        &ctx.view(),
    );
    live.stop();
    crate::diag::event(
        "boot_end",
        serde_json::json!({"stage": r.stage, "guest": r.guest, "mem": crate::diag::mem_value(&ctx.mem.mem())}),
    );
    Some((r.stage, r.guest))
}

const BOOT_RETURN_NOTE: &str = "(stage 0 = the guest ran and came back; 3 = dlopen failed, 4 = dm_host_probe_boot not found, guest -1 = it was never called)";
/// The in-process boot of `_chain` words it differently from `boot_only` (the Python has two texts).
const BOOT_RETURN_NOTE_IN_PROCESS: &str = "(stage 0 = the guest ran in this process and came back; 3 = dlopen failed, 4 = dm_host_probe_boot not found, guest -1 = never called)";

/// `boot_only`: stage 5 by itself. Returns the chain's result (0 the boot ran and came back, 1 the boot did not run: the dlopen failed or the host has no entry point, 2 no boot host,
/// 3 nothing to boot, 4 the boot was refused: the game has no shader library and this launch could not build it, 131 / 130 a pause / a cancel during that stage 3: not booted). The Python returned 0 for a boot that did not run (review r3 §5.2): a
/// caller that decides on the code took it for a game that ran.
pub fn boot_only(ctx: &mut Ctx) -> i32 {
    if ctx.adapters.boot.is_none() {
        ctx.log("boot-only: no boot host in this build (the dlopen adapter is not registered)");
        return 2;
    }
    let modules = profile_modules(ctx)
        .unwrap_or_else(|| ctx.cfg.get_or("modules", "main subsdk0 sdk").to_string());
    let framework = ctx.cfg.get_or("framework_out", "").to_string();
    let flat = ctx
        .cfg
        .get("extract_out")
        .map(PathBuf::from)
        .unwrap_or_else(|| ctx.docs.extract())
        .join("flat");
    // The profile can list a module this game's ExeFS does not carry (Mario Kart World's base .nup has no subsdk1): stage 2 dropped it, so flat/ has none of its files. Asking the
    // host for it would fail the boot the build just finished; boot what was assembled. main always stays: without it there is no game, and the host says so.
    let modules = {
        let all: Vec<&str> = modules.split_whitespace().collect();
        let (kept, gone): (Vec<&str>, Vec<&str>) = all.iter().partition(|m| {
            **m == "main" || !is_dir(&flat) || is_file(&flat.join(format!("{m}.patched.bin")))
        });
        if !gone.is_empty() {
            ctx.log(&format!(
                "boot-only: modules {} (not in flat/, the game's ExeFS has no {}: the build dropped it)",
                kept.join(" "),
                gone.join(" ")
            ));
        }
        kept.join(" ")
    };
    let profile = ctx.cfg.get_or("profile", "").to_string();
    // Nothing to boot from: the request can never be honoured until a build exists, so it goes (an armed request nobody can honour would make every launch try, and fail, by itself).
    // The next tap runs the whole chain.
    if framework.is_empty() || !is_file(Path::new(&framework)) {
        ctx.log(&format!("boot-only: no built framework at '{framework}' -- run a build first (this launch has nothing to boot)"));
        boot_only_consume(
            ctx,
            "there is no framework to boot, so a build has to come first",
        );
        return 3;
    }
    if !is_dir(&flat) {
        ctx.log(&format!("boot-only: no {} -- the framework was built from a rewrite that is no longer there; run a build", flat.display()));
        boot_only_consume(
            ctx,
            "the rewrite the framework was built from is gone, so a build has to come first",
        );
        return 3;
    }
    // Only a framework THIS app may boot is booted without a rebuild: its stamp's framework id is this app's (`framework_id`), or, for a stamp from before framework ids, its
    // build id is (`state::framework_matches`). A config with neither (a harness that predates them) skips the check; the app always writes both.
    let want = ctx.cfg.get_or("app_build", "").to_string();
    let want_fw = ctx.cfg.get_or("framework_id", "").to_string();
    if !want.is_empty() || !want_fw.is_empty() {
        let stamp = read_framework_stamp(ctx);
        let have = stamp.get_or("app_build", "").to_string();
        let have_fw = stamp.get_or("framework_id", "").to_string();
        if !framework_matches(&have, &have_fw, &want, &want_fw) {
            let id = |b: &str, f: &str| match (b.is_empty(), f.is_empty()) {
                (true, true) => "an unstamped build".to_string(),
                (false, true) => b.to_string(),
                (true, false) => format!("framework id {f}"),
                (false, false) => format!("{b} (framework id {f})"),
            };
            ctx.log(&format!(
                "boot-only: the framework on disk was built by '{}', this app is '{}' -- it would run an older host and rewrite, so it is not booted; run a build",
                id(&have, &have_fw),
                id(&want, &want_fw)
            ));
            boot_only_consume(
                ctx,
                "the framework was not built by this version of the app, so a build has to come first",
            );
            return 3;
        }
    }
    // Signed ad hoc for export only (no .p12 at the build): iOS refuses to map such a framework from Documents on a device, so it is never dlopen'd here. Export it, or sign it again.
    if framework_is_export_only(ctx) {
        ctx.log("boot-only: [RED] the framework was signed ad hoc for export only (no .p12 when it was built): it is not booted here -- Export it as its own app (the installer re-signs it), or import your .p12 and sign it again to play inside iBuilder");
        boot_only_consume(
            ctx,
            "the framework is signed for export only, so it cannot be booted here",
        );
        return 3;
    }
    ctx.log(&format!(
        "stage 5 boot-only: a process that ran no stage of the chain boots {framework}"
    ));
    ctx.log(&format!(
        "  modules='{modules}' profile={profile} romfs={}",
        ctx.cfg.get_or("romfs", "")
    ));
    release_memory("before-boot", &*ctx.log, &*ctx.mem);
    // A framework built before stage 3 could translate (or a stage 3 that did not finish) leaves a game with no library: build it here, in this launch that rewrote nothing, rather than boot
    // a game that draws nothing. A library that cannot be built refuses the boot (review r4 §4.1: its result was not looked at, and the guest started with no library).
    let built = build_shaders(ctx);
    // A pause or cancel asked for during that stage 3 ends it early with no library (`Ok(None)`, or an error the stop caused): the game is not booted either (review r4 follow-up H1: it
    // was, with no library, rc 0, the request consumed). Looked at before the error, as `chain::timed_stage` does.
    if ctx.cancelled() {
        let kind = ctx.stop_kind();
        ctx.log(&format!(
            "[stop] stage 5 boot-only: a {} was asked for during stage 3 (the shader library{}): the game is not booted",
            crate::stop::name(kind),
            match &built {
                Ok(Some(_)) => " was there",
                Ok(None) => " is not finished",
                Err(_) => " could not be finished",
            }
        ));
        if kind == crate::stop::PAUSE {
            ctx.log("[stop] boot-only: the request stays: a later launch or tap continues the library from where it stopped, then boots");
        } else {
            boot_only_consume(ctx, "the boot was cancelled during stage 3");
        }
        if fs::remove_file(ctx.docs.running()).is_ok() {
            ctx.log("the run ended (marker cleared): stopped by the user, this run was not killed");
        }
        mem_line("chain-end", &*ctx.log, &*ctx.mem);
        return if kind == crate::stop::PAUSE {
            crate::chain::RC_PAUSED
        } else {
            crate::chain::RC_CANCELLED
        };
    }
    if let Err(e) = built {
        let why = format!("failed (no library: {e})");
        ctx.log(&format!("stage 5 boot-only REFUSED: [RED] the game has no shader library and this launch could not build it ({why}): it would draw nothing, so it is not booted -- tap Play to try again, rebuild it, or send the log"));
        ctx.status.insert("shaders".into(), why);
        ctx.timing.result = "failed (shaders)".into();
        // not a boot that died: a later launch must not start it by itself again (it would be refused the same way); the framework stays, and a tap tries the library again
        boot_only_consume(
            ctx,
            "the shader library could not be built, so the game was not booted",
        );
        if fs::remove_file(ctx.docs.running()).is_ok() {
            ctx.log(
                "the run ended (marker cleared): the boot was refused, this run was not killed",
            );
        }
        mem_line("chain-end", &*ctx.log, &*ctx.mem);
        return crate::chain::RC_NO_SHADER_LIBRARY;
    }
    if let Some(tid) = ctx.title_id()
        && launcher_stale(ctx, &tid)
    {
        write_launcher_settings(ctx, &flat); // none yet, or written by an app with another settings catalogue
    }
    prepare_boot(ctx);
    write_stage(&ctx.docs, 5, 5, BOOT_STAGE_TEXT, Some(&*ctx.log));
    // The request STAYS, as `attempted`, until the boot has come back: a boot that dies in its first second leaves it for the next tap, instead of a launch with nothing to resume.
    boot_only_mark_attempted(ctx);
    let (rc, guest) = run_boot(ctx, Path::new(&framework), &flat, &modules).unwrap_or((3, -1));
    ctx.log(&format!(
        "stage 5 boot-only returned {rc}, the guest returned {guest} {BOOT_RETURN_NOTE}"
    ));
    if note_boot_result(ctx, rc) {
    } else if rc == 0 {
        boot_only_consume(ctx, "the boot came back");
    } else {
        ctx.log(&format!("boot-only: the boot did not run (stage {rc}); the request stays, attempted -- a tap boots again without a rebuild"));
    }
    if fs::remove_file(ctx.docs.running()).is_ok() {
        ctx.log(
            "the boot came back: interrupted-run marker cleared (this run did not end by a kill)",
        );
    }
    mem_line("chain-end", &*ctx.log, &*ctx.mem);
    if rc == 0 { 0 } else { crate::chain::RC_FAILED }
}

/// What the chain does at its end with a built and signed framework (`PIPELINE_BOOT=1`): the player starts the game (trap 49): a build ends in the library, with the game ready. The boot also
/// needs a process that built nothing -- right after Grizzy's 13-minute build the guest stalled before its first frame (0 fps, a black screen), while a fresh launch played at 60 fps. Unless
/// `boot_after_build=1` (the harness), stage 5 is not started; if the memory left is below `boot_floor_mib` the boot is handed to the next launch (`gameapp.boot-only`); otherwise it runs here.
pub fn boot_after_build(ctx: &mut Ctx, framework: &Path, flat: &Path) {
    let avail = ctx.mem.mem().available;
    let floor = crate::util::mib_bytes(ctx.cfg.int_or("boot_floor_mib", 2500));
    let modules = ctx.cfg.get_or("modules", "main subsdk0 sdk").to_string();
    if ctx.cfg.get_or("boot_after_build", "0") != "1" {
        ctx.log("stage 5 not started: the game is built and signed; it starts when you tap Play (boot_after_build=0)");
        release_memory("build-done", &*ctx.log, &*ctx.mem);
        if fs::remove_file(ctx.docs.running()).is_ok() {
            ctx.log("the chain finished (marker cleared): the run ends here, and the boot is its own launch");
        }
    } else if avail > 0 && avail < floor {
        let text = format!(
            "boot {} in a fresh process (in-process boot refused: {:.0} MiB available < {} MiB)\n",
            framework.display(),
            avail as f64 / 1048576.0,
            floor / 1048576
        );
        let written = crate::util::write_atomic(&ctx.docs.boot_only(), text.as_bytes());
        ctx.log(&format!("stage 5 boot NOT STARTED IN THIS PROCESS: {:.0} MiB available for the guest, below the {} MiB floor", avail as f64 / 1048576.0, floor / 1048576));
        ctx.log("  the guest commits about 2 GB of its alias region before the first frames (measured: \"alias region mapped up to +1.00 ... +2.01 GB\" around \">>> nnMain\"), so this process cannot host it with what is left");
        ctx.log(&format!("  the framework is built and SIGNED ({}): this run did all the work, and the guest is the one that needs the machine", framework.display()));
        match written {
            Ok(()) => ctx.log("  wrote Documents/gameapp.boot-only: close the app and reopen it -- the next launch boots it in a process that has rewritten nothing (~3.3 GiB free) and starts by itself"),
            Err(e) => ctx.log(&format!("  could not write {}: {e} -- the next launch will NOT boot it by itself; tap Play to boot the framework that is already built", ctx.docs.boot_only().display())),
        }
        release_memory("at-refusal", &*ctx.log, &*ctx.mem);
        if fs::remove_file(ctx.docs.running()).is_ok() {
            ctx.log("the chain finished (marker cleared): the run ends here, and the boot is its own launch");
        }
    } else {
        let seconds = ctx.cfg.int_or("boot_seconds", 45);
        ctx.log(&format!(
            "stage 5 boot (in-process): boot({}, {}, '{modules}', {}, {}, {seconds})",
            framework.display(),
            flat.display(),
            ctx.cfg.get_or("profile", ""),
            ctx.cfg.get_or("romfs", "")
        ));
        release_memory("before-boot", &*ctx.log, &*ctx.mem);
        prepare_boot(ctx);
        write_stage(&ctx.docs, 5, 5, BOOT_STAGE_TEXT, Some(&*ctx.log));
        let (rc, guest) = run_boot(ctx, framework, flat, &modules).unwrap_or((3, -1));
        // The boot RETURNING is the only in-process evidence of an end that let the chain finish. A run whose process is killed while the guest runs (jetsam, watchdog) never gets here, so the
        // interrupted-run marker must STILL BE THERE: the next launch has to say "the guest was running when this process ended, and here is how much memory it had", not start the chain over
        // (trap 18).
        ctx.log(&format!(
            "stage 5 boot returned {rc}, the guest returned {guest} {BOOT_RETURN_NOTE_IN_PROCESS}"
        ));
        note_boot_result(ctx, rc);
        if fs::remove_file(ctx.docs.running()).is_ok() {
            ctx.log("the boot came back: interrupted-run marker cleared (this run did not end by a kill)");
        }
    }
}
