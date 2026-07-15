# Contributing to x265

Thanks for your interest in improving x265, the open-source HEVC/H.265 video
encoder. This guide describes how to build the project, the coding conventions
we follow, and how to get a change reviewed and merged.

x265 is dual-licensed under the [GNU GPL v2](COPYING) and a
[commercial license](http://x265.org). To keep this dual-licensing possible,
**all contributors must sign a Contributor License Agreement (CLA) before their
patches can be accepted** — see [Contributor License Agreement](#contributor-license-agreement)
below.

This document consolidates the guidance previously spread across the
[developer wiki](https://github.com/Multicorewareinc/x265/wiki/Contribute).
The wiki page remains available and is kept in sync with this file.

## Table of contents

- [Contributor License Agreement](#contributor-license-agreement)
- [Where development happens](#where-development-happens)
- [Reporting bugs](#reporting-bugs)
- [Building from source](#building-from-source)
- [Running the tests](#running-the-tests)
- [Coding style](#coding-style)
- [Commit guidelines](#commit-guidelines)
- [Submitting a change](#submitting-a-change)
  - [Option A: GitHub pull request](#option-a-github-pull-request)
  - [Option B: Mailing-list patches](#option-b-mailing-list-patches)
- [Changing the public API](#changing-the-public-api)
- [Contributing assembly / SIMD code](#contributing-assembly--simd-code)
- [Legal / licensing](#legal--licensing)

## Contributor License Agreement

Contributing to x265 requires a signed Contributor License Agreement. This is a
**one-time** requirement per contributor.

1. Download the agreement:
   <https://github.com/Multicorewareinc/x265/releases/download/3.3/x265ContributorAgreement.pdf>
2. Sign it and return it using the method described in the document.

Because x265 is distributed under both the GNU GPL v2 and a commercial license,
the CLA is what allows your contribution to be shipped under both. Patches from
contributors who have not signed the CLA cannot be merged.

## Where development happens

- **Source & pull requests:** <https://github.com/Multicorewareinc/x265>
- **Issue tracker:** <https://github.com/Multicorewareinc/x265/issues>
- **Developer wiki:** <https://github.com/Multicorewareinc/x265/wiki>
- **Documentation:** <http://x265.readthedocs.org/en/master/>
- **Development mailing list:** `x265-devel@videolan.org`
  ([subscribe](http://mailman.videolan.org/listinfo/x265-devel))
- **Commits mailing list:** `x265-commits@videolan.org`
- **Patch review (Patchwork):** <http://patches.videolan.org/project/x265-devel/list/>
- **IRC:** `#x265` on Libera.Chat

Both GitHub pull requests and mailing-list patches are accepted. As a rule of
thumb: everyday changes, CI/docs, and single patches are easiest as a
[GitHub PR](#option-a-github-pull-request); multi-part patch series and
provenance-sensitive work fit the [mailing-list flow](#option-b-mailing-list-patches),
which is the long-standing VideoLAN-project convention.

## Reporting bugs

Before opening an issue, please:

1. Check whether the problem still reproduces on the tip of `master`.
2. Search [existing issues](https://github.com/Multicorewareinc/x265/issues) to
   avoid duplicates.

A good report includes:

- The x265 version (`x265 --version`) and how it was built (compiler, OS,
  target architecture, 8/10/12-bit or multilib).
- The **exact command line** and, where possible, a small input clip or a
  description of how to generate one.
- What you expected to happen and what actually happened (including any crash
  backtrace or assertion text).

Because a video encoder routinely parses untrusted input, please read
[SECURITY.md](SECURITY.md) before filing anything that looks like a memory-safety
issue (crash, out-of-bounds read/write, hang on crafted input). Do **not** open a
public issue for those — report them privately.

## Building from source

Always build from a clone of the repository — **never develop from a downloaded
source tarball**, as it lacks the version metadata and history the tooling
relies on:

```sh
git clone https://github.com/Multicorewareinc/x265
```

x265 uses CMake. Ready-made generator scripts live under `build/` for common
targets (`build/linux`, `build/aarch64-linux`, `build/riscv64-linux`,
`build/vc17-x86_64`, and others — see `build/README.txt`).

### Linux / macOS

```sh
cd build/linux
cmake -G "Unix Makefiles" ../../source
make -j"$(nproc)"
```

For an interactive configuration menu, run `ccmake ../../source` (or use the
provided `build/linux/make-Makefiles.bash`).

### Windows (MSVC)

Use one of the `build/vc*` folders and the matching `.bat` script, or point the
CMake GUI at `source/`. NASM is required for x86 assembly.

### High-bit-depth and multilib builds

The default build is 8-bit. To build 10-bit or 12-bit, configure with:

```sh
cmake -DHIGH_BIT_DEPTH=ON -DMAIN12=OFF ../../source   # 10-bit
cmake -DHIGH_BIT_DEPTH=ON -DMAIN12=ON  ../../source   # 12-bit
```

A combined 8+10+12-bit "multilib" build is produced by
`build/linux/multilib.sh`. When adding or changing a feature, please make sure
it still builds in **all** of these configurations — CI verifies each of them.

## Running the tests

x265 ships its own correctness harness, `TestBench`, which validates the
optimized (SIMD/assembly) primitives against the C reference implementations.
Enable it at configure time:

```sh
cmake -DENABLE_TESTS=ON ../../source
make -j"$(nproc)"
./test/TestBench
```

`TestBench` must pass on every architecture your change affects. For
functional/regression coverage there are additional test descriptors under
`source/test/` (`regression-tests.txt`, `rate-control-tests.txt`,
`save-load-tests.txt`, `smoke-tests.txt`, …) that drive the CLI over a corpus of
clips; the CI workflow (`.github/workflows/ci.yml`) runs these across presets,
rate-control modes, threading options, and bit depths.

Please run at least the smoke tests and the relevant `TestBench` primitives
locally before submitting a change.

## Coding style

The house style is:

- 4-space indentation, **no tabs**.
- [Allman](https://en.wikipedia.org/wiki/Indentation_style#Allman_style) braces
  (opening brace on its own line).
- 120-column soft limit.
- LLVM base conventions otherwise.

These rules are captured in [`.clang-format`](.clang-format) at the repository
root, and the CI "Code Quality" job checks the lines you change against them
(plus `cppcheck` for warnings, performance, portability and style). The older
`doc/uncrustify/codingstyle.cfg` describes the same conventions.

To format only the lines you touched:

```sh
git clang-format origin/master
```

**Do not reformat unrelated code, and never mix whitespace changes with logic
changes.** Keep each diff limited to the lines your change actually needs —
whitespace-only churn in surrounding code makes reviews and `git blame` harder
and will be asked to be reverted.

## Commit guidelines

- The first line is a brief summary. Add a blank line and a fuller explanation
  below it when the change isn't self-evident.
- **Small, coherent commits are better than one large commit.** Group logically
  independent changes into separate commits; use `git stash` to shelve unrelated
  work in progress.
- Where it helps the reader, prefix the summary with the area touched, matching
  existing history — e.g. `AArch64:`, `LoongArch64:`, `RISC-V:`, `CMake:`,
  `x86:`, `ci:`, or a `fix:` / `chore:` / `docs:` tag for small changes.
- Reference issues with `#<number>`.

Examples drawn from the project's history:

```
AArch64: Improve DCT16 and DCT32 Neon implementations
CMake: Fix building on PowerPC64 with Clang
fix: scenecut not active when bframes = 0
```

## Submitting a change

First make sure your tree is clean and current:

```sh
git status                 # commit or `git reset --hard` any stray changes
git fetch && git pull      # sync with the latest master
git checkout master        # (or your up-to-date topic branch)
```

Then choose one of the two submission paths below.

### Option A: GitHub pull request

1. Fork the repository (external contributors) or create a topic branch
   (maintainers). Branch from an up-to-date `master`.
2. Keep the PR focused on a single logical change.
3. Make sure the change **builds in 8-bit, 10-bit, 12-bit and multilib**, and
   that `TestBench` passes for every architecture you touched.
4. Run `git clang-format origin/master` so the "Code Quality" check passes.
5. Open the PR against `master`. The CI workflow will build all bit-depth
   configurations and run the CLI/regression/threading test suites; a green run
   is expected before review.
6. Address review feedback by pushing follow-up commits (the maintainers will
   squash if needed).

### Option B: Mailing-list patches

This is the traditional VideoLAN workflow and is well suited to patch series.

1. Confirm your commits are present: `git log`.
2. Generate patch files, one per commit, into a directory:

   ```sh
   git format-patch master -o patches
   # or a specific commit:
   git format-patch -1 <commit-sha> -o patches
   ```

3. Flag the patches with the target branch in the subject, e.g. `[MASTER]`.
4. Email the generated patches to **`x265-devel@videolan.org`** (you must be
   subscribed to post). Patches are tracked in
   [Patchwork](http://patches.videolan.org/project/x265-devel/list/).

## Changing the public API

If your change adds or alters anything in the public API, extra steps are
**required** so the library stays consistent and ABI changes are tracked.

When you add a new `x265_param` member, you must:

1. Document the member in `source/x265.h`.
2. Parse it in `x265_param_parse()` (`source/common/param.cpp`).
3. Emit it in `x265_param2string()` (`source/common/param.cpp`).
4. Add `getopt()` handling and CLI `--help` text in `source/x265cli.h`
   (and `source/x265cli.cpp`).
5. Document the CLI option in `doc/reST/cli.rst`.
6. Increment `X265_BUILD` in `source/CMakeLists.txt` (the soname / API build
   number — currently `216`).
7. Add coverage to `source/test/smoke-tests.txt` and
   `source/test/regression-tests.txt`.

The same expectations (documentation, `X265_BUILD` bump, and tests) apply when
you add or change any public function or structure.

## Contributing assembly / SIMD code

Most recent optimization work targets AArch64 (Neon/SVE/SVE2), LoongArch64,
RISC-V and x86. When adding or changing an optimized primitive:

- Provide a scalar C reference (or reuse the existing one) so the primitive can
  be validated.
- Add or extend the corresponding harness in `source/test/` so `TestBench`
  exercises the new code path, and confirm it reports a match against the C
  reference.
- Wire the primitive into the per-architecture dispatch/setup so it is only
  selected when the required CPU features are present.
- Note any measured speedup (clip, CPU, before/after) in the submission.

## Legal / licensing

x265 is distributed under **both** the GNU GPL v2 and a commercial license. To
keep this dual-licensing possible, contributions must be your own work (or
work you are authorized to submit) and licensable under both, and you must have
a signed [CLA](#contributor-license-agreement) on file. Do not paste code from
GPL-incompatible or unknown-provenance sources. If your employer owns your work,
make sure you have permission to contribute it.

By submitting a contribution you assert that it can be distributed under the
terms in [COPYING](COPYING) and under the x265 commercial license.
