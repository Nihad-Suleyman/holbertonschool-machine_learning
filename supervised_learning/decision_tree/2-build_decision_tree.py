def left_child_add_prefix(self, text):
    """Add left-branch prefix formatting."""
    lines = text.split("\n")
    out = "+--" + lines[0] + "\n"
    for line in lines[1:]:
        out += "| " + line + "\n"
    return out

def right_child_add_prefix(self, text):
    """Add right-branch prefix formatting."""
    lines = text.split("\n")
    out = "+--" + lines[0] + "\n"
    for line in lines[1:]:
        out += "  " + line + "\n"
    return out
