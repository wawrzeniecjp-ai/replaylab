"""The experiment harness: play agents and scripted policies on the same seeds, record, compare.

May import anything; nothing imports it. Start with ``runner.py:open_episode`` (everything
one episode needs) and ``llm_player.py:LLMPlayer`` (how an agent plays a round). The game
loop itself is ``engine/episode.py:Episode.play``.

    cli.py          the replaylab-eval commands, one function each
    setup.py        what a command sets up before its first episode, and every refusal
    runner.py       one episode (open_episode, play_episode) and the matrix of them:
                    resumable, concurrent, recorded
    llm_player.py   the LLM player: model turns and tool calls, round by round
    mcp_bridge.py   MCP tools <-> OpenAI-style function calls
    llm*.py         provider adapters (OpenAI-compatible, Anthropic)
    spend.py        prices and the spend guard; models.py: config/models.yaml
    records.py      episodes.jsonl; seeds.py: episode_seed(game, trial)
    final_run.py    the run-once lock of the final test
    ladder.py       the ladder report: every scripted policy on many seeds
    stats.py        paired bootstrap; scorers.py: per-record metrics
"""
