import logging
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from io import StringIO

log = logging.getLogger(__name__)
def log_color(content, title=""):
    """Enhanced logging with colored console output for main process."""
    try:
        console = Console()
        console.print(Panel(content, title=title, border_style="cyan", title_align="left"))

        # Log the message as plain text for log files
        string_io = StringIO()
        plain_console = Console(file=string_io, highlight=False)
        plain_console.print(Panel(content, title=title, border_style="none", title_align="left"))
        log.info("\n" + string_io.getvalue())
    except Exception as e:
        # Fallback to plain text logging if Console fails
        log.info(f"Error logging content: {e}")