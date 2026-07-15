# Security Policy

x265 is an HEVC/H.265 encoder that routinely processes untrusted input —
YUV/Y4M source frames, command-line and config-file parameters, and, for some
features, auxiliary data such as HDR10+ metadata. Memory-safety and
denial-of-service issues in these paths can have security impact for the many
applications that embed `libx265`. We take such reports seriously and appreciate
responsible disclosure.

## Supported versions

Security fixes are developed against the tip of the `master` branch and land in
the next release. The most recent release is the primary supported version; the
current version is recorded in [`x265Version.txt`](x265Version.txt). Older
releases are generally not patched in place — where feasible, please confirm an
issue still reproduces on `master` before reporting.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately using either of the following:

1. **Email (preferred):** contact the maintainers at
   `x265@multicorewareinc.com`. You may encrypt sensitive details on request.
2. **GitHub private advisories:** if private vulnerability reporting is enabled
   for this repository, you can open a report from the **Security** tab
   ("Report a vulnerability"). If that option is not available, use email
   instead.

For coordinated disclosure that affects the broader ecosystem, the VideoLAN
security contact (`security@videolan.org`) can also be used, as x265 is
distributed via VideoLAN.

### What to include

A useful report contains:

- The x265 version (`x265 --version`) and build configuration (compiler, OS,
  target architecture, 8/10/12-bit or multilib).
- The **exact command line** used.
- A **minimal reproducer**: the crafted input file and/or parameters that
  trigger the issue. Keep sample inputs as small as possible.
- The observed behavior — crash, hang, out-of-bounds read/write, excessive
  memory/CPU use — with a stack trace or sanitizer output if you have one
  (builds with `-fsanitize=address,undefined` are especially helpful).
- Your assessment of impact, if any.

### In scope

- Memory-safety defects reachable from encoder input or parameters: buffer
  overflows/underflows, use-after-free, out-of-bounds access, integer overflows
  leading to unsafe allocation or indexing.
- Crashes, assertion failures, or unbounded resource consumption triggered by
  crafted input or parameter combinations.
- Issues in the CLI or in auxiliary parsers (e.g. HDR10+ metadata handling).

### Out of scope

- Bugs with no security impact — please file those as normal
  [issues](https://github.com/Multicorewareinc/x265/issues).
- Vulnerabilities in third-party dependencies, downstream distributions, or
  applications that merely embed `libx265` (report those to the relevant
  project).
- Encode quality, performance, or standards-conformance concerns.

## Disclosure process

- We will acknowledge your report, ordinarily within a few business days.
- We will investigate, keep you informed of progress, and coordinate on a fix
  and a disclosure timeline.
- With your consent, we are glad to credit you in the advisory and release
  notes once a fix ships.

Thank you for helping keep x265 and its users safe.
