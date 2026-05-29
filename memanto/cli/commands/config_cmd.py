"""
MEMANTO CLI - Config commands (show).
"""

from rich.table import Table

from memanto.cli.commands._shared import (
    config_app,
    config_manager,
    console,
)


@config_app.command("show")
def config_show():
    """Display current configuration."""
    api_key = config_manager.get_api_key()
    server_cfg = config_manager.get_server_config()
    cli_cfg = config_manager.get_cli_config()
    ans_cfg = config_manager.get_answer_config()
    rec_cfg = config_manager.get_recall_config()
    active_agent_id, active_session_token = config_manager.get_active_session()
    schedule_time = config_manager.get_schedule_time()

    table = Table(title="MEMANTO Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Config Dir", str(config_manager.config_dir))
    table.add_row("Server URL", f"{server_cfg['url']}:{server_cfg['port']}")
    table.add_row("API Key", "***configured***" if api_key else "not set")
    table.add_row("Active Agent", active_agent_id or "none")
    table.add_row("Session Active", "yes" if active_session_token else "no")
    table.add_row("Schedule Time", schedule_time)
    table.add_row("Interactive Mode", str(cli_cfg.get("interactive_mode", True)))
    table.add_row("Smart Parse", str(cli_cfg.get("smart_parse", True)))

    # Answer Config
    table.add_section()
    table.add_row("[bold]Answer Config[/bold]", "")
    table.add_row("  Model", ans_cfg.get("model", "—"))
    table.add_row("  Temperature", str(ans_cfg.get("temperature", 0.7)))
    table.add_row("  Limit", str(ans_cfg.get("answer_limit", 5)))
    table.add_row("  Threshold", str(ans_cfg.get("threshold", 0.25)))

    # Recall Config
    table.add_section()
    table.add_row("[bold]Recall Config[/bold]", "")
    table.add_row("  Limit (Top N)", str(rec_cfg.get("limit", 10)))
    table.add_row("  Min Similarity", str(rec_cfg.get("min_similarity", 0.0)))

    console.print(table)
