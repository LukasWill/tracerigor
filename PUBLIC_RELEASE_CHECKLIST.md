# Public release checklist

Before publishing a snapshot:

- [ ] Confirm the public repository URL and add it to package metadata.
- [ ] Add final project authors and an archival identifier to citation metadata.
- [ ] Run the validation commands documented in the release report.
- [ ] Review `git status` and ensure the cleanup quarantine is not staged.
- [ ] Scan the complete Git history for revoked credentials and private artifacts.
- [ ] Publish from a fresh repository/history if internal development history must
      not be exposed.
- [ ] Confirm that no model checkpoints, raw runs, judge-case exports, or local
      datasets are staged.
- [ ] Verify licences for any optional environment assets added after this cleanup.
- [x] Exclude research-working `analysis/` and `paper/` directories from the
      software release.

The working-tree cleanup does not remove values from existing Git history. Any
credential ever committed must be revoked independently of repository cleanup.
The release must be initialized from the reviewed working snapshot rather than
from this repository's existing Git ancestry.
