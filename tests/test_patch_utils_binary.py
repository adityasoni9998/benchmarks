"""remove_binary_diffs drops binary file blocks and keeps every other byte."""

from benchmarks.utils.patch_utils import remove_binary_diffs


TEXT = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " x = 1\n"
    "-y = 2\n"
    "+y = 3\n"
)
BINARY = (
    "diff --git a/data.pickle b/data.pickle\n"
    "new file mode 100644\n"
    "index 0000000000..7576f63c01\n"
    "Binary files /dev/null and b/data.pickle differ\n"
)
GIT_BINARY = (
    "diff --git a/img.png b/img.png\n"
    "new file mode 100644\n"
    "index 0000000..1111111\n"
    "GIT binary patch\n"
    "literal 3\n"
    "Kcmb1ce$\n"
    "\n"
)


def test_text_only_patch_is_unchanged_including_final_newline():
    assert remove_binary_diffs(TEXT) == TEXT


def test_binary_blocks_are_removed_anywhere():
    assert remove_binary_diffs(BINARY + TEXT) == TEXT
    assert remove_binary_diffs(TEXT + BINARY) == TEXT
    assert remove_binary_diffs(TEXT + GIT_BINARY + TEXT) == TEXT + TEXT


def test_crlf_lines_are_kept():
    crlf = TEXT.replace(" x = 1\n", " x = 1\r\n").replace("+y = 3\n", "+y = 3\r\n")
    assert remove_binary_diffs(crlf + BINARY) == crlf


def test_text_mentioning_binary_files_is_kept():
    mention = TEXT.replace("+y = 3\n", '+msg = "Binary files differ"\n')
    assert remove_binary_diffs(mention) == mention


def test_empty_and_non_diff_input():
    assert remove_binary_diffs("") == ""
    assert remove_binary_diffs("not a diff\n") == "not a diff\n"
