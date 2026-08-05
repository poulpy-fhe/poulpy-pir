# Security Policy

`poulpy-pir` implements cryptographic protocols and should be treated as
security-sensitive software.

## Supported Versions

Security fixes are currently provided for the latest released `0.x` version.
Because the crate is pre-1.0, public APIs and parameter defaults may still
change between minor releases.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the maintainers before
opening a public issue. If GitHub private vulnerability reporting is enabled for
the repository, use that channel. Otherwise, contact the maintainers through the
repository owner profile and include:

- affected version or commit
- a concise description of the issue
- reproduction steps or proof-of-concept input, if available
- expected impact

We will acknowledge reports as quickly as practical, investigate, and coordinate
public disclosure after a fix or mitigation is available.

## Audit Status

This crate has not yet had an independent third-party security audit. Do not
present it as audited software unless that changes and the audit report is
published.

## Deployment Notes

- Treat query and response bytes as untrusted network input.
- Use the fallible `try_*` APIs at service boundaries.
- Keep server-side rate limits and request-size limits in the surrounding
  transport layer.
- Validate keyword-PIR records client-side against the requested key, as the
  MPHF maps out-of-set keys to valid-looking indices.
- Revisit parameter choices and side-channel exposure for each deployment.

