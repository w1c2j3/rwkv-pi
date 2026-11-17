"""Tool -> category mapping for the Pi side (shared-format process validation).

Validation is by category, never by tool name. The DSH side keeps its own table.
Stock Pi (Harbor's default) exposes only read/bash/edit/write; listing and search
then happen through bash and count as command execution.
"""
FILE_READ = "file_read"
CONTENT_SEARCH = "content_search"
FILE_LISTING = "file_listing"
COMMAND_EXECUTION = "command_execution"
FILE_MODIFICATION = "file_modification"
TASK_PLANNING = "task_planning"
WEB_BROWSING = "web_browsing"
CATEGORIES = (FILE_READ, CONTENT_SEARCH, FILE_LISTING, COMMAND_EXECUTION, FILE_MODIFICATION, TASK_PLANNING, WEB_BROWSING)

PI_TOOLS = {
    "read": FILE_READ,
    "grep": CONTENT_SEARCH,
    "find": FILE_LISTING,
    "ls": FILE_LISTING,
    "bash": COMMAND_EXECUTION,
    "edit": FILE_MODIFICATION,
    "write": FILE_MODIFICATION,
    "todo_plan": TASK_PLANNING,
    "todo_update": TASK_PLANNING,
    "web_search": WEB_BROWSING,
    "web_fetch": WEB_BROWSING,
}

# Categories with no built-in stock Pi tool unless host web tools are registered.
PI_MISSING = (WEB_BROWSING,)


def category(tool_name: str) -> str | None:
    return PI_TOOLS.get(tool_name)
