//! Stage 2: the rewrite of each module, with the cache (`rewrite_key`, `rewrite_cache_dir`, `do_rewrite`, `rewrite_is_cached`).
//!
//! A module whose `rewrite_key` matches the one cached in `Documents/games/<id>/rewrite-cache/<m>` is taken from there (about 150 s for a large main on the phone with the
//! CPython rewriter: every repeat of the chain, after a failure or a heap-check-free rebuild, paid it again). The key: the module's bytes, this build of the app (the rewriter and
//! the engine come with it), the game's profile tables (the heap check edits them) and the digest of the exefs patches of the enabled mods that name the module (`mods.plan`; "" adds
//! nothing, so a game without mods keeps its cache).

use crate::adapters::RewriteRequest;
use crate::ctx::Ctx;
use crate::error::{PipelineError, Result};
use crate::memory::{release_memory, rewrite_jobs_now};
use crate::mods::{finish_rewrite, mods_plan};
use crate::profile::table_dir;
use crate::stage::write_stage;
use crate::util::{is_dir, is_file, link_tree, list_names, read_text, rmtree};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Which version of the rewriter a cached rewrite belongs to: the config's `framework_id` when it has one (the build script's hash of the stage crates, the host and the tables:
/// the rewriter's code is part of it, so the same id and the same inputs give byte-identical output, and an app update that keeps the framework id keeps the cache too), else
/// its `app_build` (a config from before framework ids: the key is then byte for byte what it always was). `None`: neither, no caching.
fn rewriter_version(ctx: &Ctx) -> Option<(&'static str, &str)> {
    match ctx.cfg.get_or("framework_id", "") {
        "" => match ctx.cfg.get_or("app_build", "") {
            "" => None,
            b => Some(("app_build", b)),
        },
        f => Some(("framework_id", f)),
    }
}

/// `rewrite_key(cfg, src, mods_digest)`: what a module's rewrite depends on. `Ok(None)` (no caching) without a `framework_id` or an `app_build` (`rewriter_version`).
/// `mods_digest` is the digest `mods.plan` made for this module ("" for none: the key is then what it was before mods existed).
pub fn rewrite_key(ctx: &Ctx, src: &Path, mods_digest: &str) -> std::io::Result<Option<String>> {
    let Some((label, version)) = rewriter_version(ctx) else {
        return Ok(None);
    };
    let mut h = Sha256::new();
    h.update(label.as_bytes());
    h.update(b"=");
    h.update(version.as_bytes());
    h.update(b"\0");
    if !mods_digest.is_empty() {
        h.update(b"mods=");
        h.update(mods_digest.as_bytes());
        h.update(b"\0");
    }
    let mut f = fs::File::open(src)?;
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    if let Some(tid) = ctx.title_id()
        && let Some(d) = table_dir(ctx, &tid).filter(|d| is_dir(d))
    {
        for name in list_names(&d) {
            let path = d.join(&name);
            if name.ends_with(".tsv") && is_file(&path) {
                h.update(b"\0");
                h.update(name.as_bytes());
                h.update(b"\0");
                h.update(&fs::read(&path)?);
            }
        }
    }
    Ok(Some(hex(&h.finalize())))
}

/// `rewrite_cache_dir`: `Documents/games/<id>/rewrite-cache/<m>`
pub fn rewrite_cache_dir(ctx: &Ctx, module: &str) -> Option<PathBuf> {
    let tid = ctx.title_id()?;
    Some(ctx.docs.game_dir(&tid).join("rewrite-cache").join(module))
}

/// The directory the rewrite writes into (`patched_dir`, else `<extract>/rewrite`).
pub fn patched_dir(ctx: &Ctx) -> PathBuf {
    match ctx.cfg.get_nonempty("patched_dir") {
        Some(p) => PathBuf::from(p),
        None => Path::new(ctx.cfg.get_or("extract_out", "")).join("rewrite"),
    }
}

/// `do_rewrite`: stage 2, memory around each module. Returns the folder the modules were written under.
pub fn do_rewrite(ctx: &mut Ctx) -> Result<PathBuf> {
    let modules = ctx.cfg.modules();
    let exe_dir = PathBuf::from(
        ctx.cfg
            .get("exe_dir")
            .ok_or_else(|| PipelineError::missing("exe_dir is not in the config"))?,
    );
    let out_dir = patched_dir(ctx);
    fs::create_dir_all(&out_dir)
        .map_err(|e| PipelineError::io(format!("create {}", out_dir.display()), e))?;
    ctx.log(&format!(
        "stage 2 rewrite: {} modules from {}",
        modules.len(),
        exe_dir.display()
    ));
    // what this ExeFS is, before any module is read for minutes: a 32-bit program, or a build the profile does not know (see preflight.rs)
    let game_tsv = ctx
        .title_id()
        .and_then(|tid| table_dir(ctx, &tid))
        .and_then(|d| read_text(&d.join("game.tsv")).ok());
    for f in crate::preflight::check(&exe_dir, &modules, game_tsv.as_deref()) {
        ctx.log(&format!(
            "preflight {}: {}",
            if f.fatal { "REFUSED" } else { "warning" },
            f.text
        ));
        if f.fatal {
            return Err(PipelineError::failed(f.text));
        }
    }
    release_memory("before-rewrite", &*ctx.log, &*ctx.mem); // the profile stage's buffers go back first: stage 2 starts from what the process really holds
    // exefs patches (IPS, pchtxt) of the enabled mods, by build id
    let modplan = mods_plan(ctx, &exe_dir, &modules, false)?;
    let entry_of = |m: &str| modplan.iter().find(|(n, _)| n == m).map(|(_, e)| e);
    let mut no_input: Vec<String> = Vec::new();
    for (i, m) in modules.iter().enumerate() {
        if ctx.cancelled() {
            return Err(PipelineError::Cancelled);
        }
        ctx.thermal.wait(3, "stage 2 rewrite, between modules"); // critical heat: the next module waits
        write_stage(
            &ctx.docs,
            2,
            5,
            &format!("rewriting the modules ({}/{}: {m})", i + 1, modules.len()),
            Some(&*ctx.log),
        );
        let src = exe_dir.join(m);
        if !is_file(&src) {
            ctx.log(&format!("  {m}: no input {} -- skipping", src.display()));
            no_input.push(m.clone());
            continue;
        }
        let sub = out_dir.join(m);
        let cache = if ctx.cfg.not_zero("rewrite_cache", "1") {
            rewrite_cache_dir(ctx, m)
        } else {
            None
        };
        let mut key = None;
        if cache.is_some() {
            match rewrite_key(
                ctx,
                &src,
                entry_of(m).map(|e| e.digest.as_str()).unwrap_or(""),
            ) {
                Ok(k) => key = k,
                Err(e) => ctx.log(&format!("  {m}: rewrite cache unusable ({e})")),
            }
        }
        if let (Some(key), Some(cache)) = (&key, &cache)
            && is_file(&cache.join("key"))
        {
            let cached = read_text(&cache.join("key")).unwrap_or_default();
            if cached.trim() == key
                && is_file(&cache.join("files").join(format!("{m}.patched.bin")))
            {
                link_tree(&cache.join("files"), &sub)
                    .map_err(|e| PipelineError::io(format!("link the cached rewrite of {m}"), e))?;
                ctx.log(&format!(
                    "  {m}: rewrite taken from the cache (same module, app build and profile)"
                ));
                continue;
            }
        }
        fs::create_dir_all(&sub)
            .map_err(|e| PipelineError::io(format!("create {}", sub.display()), e))?;
        // the rewrite must see the patched code: a patched copy replaces the module as input
        let mut src = src;
        if let (Some(entry), Some(host), Some(tid)) =
            (entry_of(m), ctx.adapters.mods.clone(), ctx.title_id())
            && let Some(p) = host
                .apply_module(ctx.docs.root(), &tid, &exe_dir, m, entry, &*ctx.log)
                .map_err(|f| PipelineError::failed(format!("mods: {f}")))?
        {
            src = p;
        }
        // only the big module is worth threads
        let jobs = if m == "main" || modules.len() == 1 {
            rewrite_jobs_now(&ctx.cfg, &*ctx.log, &*ctx.mem, &ctx.thermal)
        } else {
            1
        };
        if jobs > 1 {
            ctx.log(&format!("  {m}: rewritten on {jobs} threads (DM_REWRITE_JOBS, paches/engine/0301: the same output on any count)"));
            if let Some(l) = crate::sys::qos::log_line(2) {
                ctx.log(&format!("  {l}"));
            }
        }
        let t0 = Instant::now();
        let rewriter = ctx.adapters.rewriter.clone();
        if let Err(e) = rewriter.rewrite(
            &RewriteRequest {
                src: &src,
                out_dir: &sub,
                module: m,
                jobs,
            },
            &ctx.view(),
        ) {
            // a parser's own words ("KeyError: 2") mean nothing to a player: say what they usually are
            return Err(match &e {
                PipelineError::Failed(text) => match crate::preflight::rewrite_hint(text) {
                    Some(hint) => PipelineError::failed(format!("{m}: {text} -- {hint}")),
                    None => e,
                },
                _ => e,
            });
        }
        ctx.rewrite_ran.push(m.clone());
        let got = out_dir.join(m).join(format!("{m}.patched.bin"));
        ctx.log(&format!(
            "  {m}: rc=0 {} ({:.1}s)",
            if is_file(&got) {
                format!("wrote {}", got.display())
            } else {
                "NO OUTPUT".to_string()
            },
            t0.elapsed().as_secs_f64()
        ));
        if let (Some(key), Some(cache)) = (&key, &cache)
            && is_file(&got)
        {
            let go = || -> std::io::Result<()> {
                rmtree(cache);
                link_tree(&sub, &cache.join("files"))?;
                fs::write(cache.join("key"), format!("{key}\n")) // last: a half-written cache has no key
            };
            if let Err(e) = go() {
                ctx.log(&format!("  {m}: rewrite not cached ({e})"));
                rmtree(cache);
            }
        }
        release_memory(&format!("after-rewrite-{m}"), &*ctx.log, &*ctx.mem);
    }
    // A module the profile lists but this game's ExeFS does not carry (a .nup, a base game without its update, a build the profile was not derived from) was skipped above;
    // left in `modules`, stage 4a's make_flat refused the whole build with "the rewrite wrote no <m>" after a full rewrite and shader pass. It cannot be loaded anyway:
    // drop it here, so the assemble and the boot both use the modules that exist. main is the exception -- without it there is no game.
    if !no_input.is_empty() {
        if no_input.iter().any(|m| m == "main") {
            return Err(PipelineError::missing(format!(
                "this game's ExeFS has no main in {}",
                exe_dir.display()
            )));
        }
        let kept: Vec<String> = modules
            .iter()
            .filter(|m| !no_input.contains(m))
            .cloned()
            .collect();
        ctx.log(&format!(
            "modules: {} (dropped {}: listed by the profile, not in this game's ExeFS)",
            kept.join(" "),
            no_input.join(" ")
        ));
        ctx.cfg.set("modules", kept.join(" "));
    }
    finish_rewrite(ctx)?;
    Ok(out_dir)
}

/// `rewrite_is_cached`: every module of the rewrite is in the rewrite cache for this rewriter (`rewriter_version`) and these inputs (`do_rewrite` would take them all from it): a launch that
/// comes back for stage 3 then does not pay the rewrite again.
pub fn rewrite_is_cached(ctx: &Ctx) -> bool {
    if !ctx.cfg.not_zero("rewrite_cache", "1") || rewriter_version(ctx).is_none() {
        return false;
    }
    let modules = ctx.cfg.modules();
    // the patches' digests are part of the key; a plan that fails (an exception the Python lets through) is "not cached": stage 2 would fail on it first
    let Ok(modplan) = mods_plan(
        ctx,
        Path::new(ctx.cfg.get_or("exe_dir", "")),
        &modules,
        true,
    ) else {
        return false;
    };
    for m in modules {
        let src = Path::new(ctx.cfg.get_or("exe_dir", "")).join(&m);
        if !is_file(&src) {
            continue;
        }
        let Some(cache) = rewrite_cache_dir(ctx, &m) else {
            return false;
        };
        let digest = modplan
            .iter()
            .find(|(n, _)| *n == m)
            .map(|(_, e)| e.digest.as_str())
            .unwrap_or("");
        match rewrite_key(ctx, &src, digest) {
            Ok(Some(key)) => match read_text(&cache.join("key")) {
                Ok(c) if c.trim() == key => {}
                _ => return false,
            },
            _ => return false,
        }
        if !is_file(&cache.join("files").join(format!("{m}.patched.bin"))) {
            return false;
        }
    }
    true
}
