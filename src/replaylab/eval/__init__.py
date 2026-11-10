"""The experiment harness: play agents and scripted policies on the same seeds, record, compare.

May import anything; nothing imports it. Start with ``runner.py:run_llm_episode`` (one agent
episode, end to end) and ``loop.py:run_episode`` (the agent's turns).

    cli.py          the replaylab-eval commands, one function each
    setup.py        what a command sets up before its first episode, and every refusal
    runner.py       the games x trials x conditions matrix: resumable, concurrent, recorded
    loop.py         one agent episode: model call, tool calls, repeat, with hard limits
    mcp_bridge.py   MCP tools <-> OpenAI-style function calls
    llm*.py         provider adapters (OpenAI-compatible, Anthropic)
    spend.py        prices and the spend guard; models.py: config/models.yaml
    records.py      episodes.jsonl; seeds.py: episode_seed(game, trial)
    final_run.py    the run-once lock of the final test
    scripted.py     scripted policies on the agents' seeds; ladder.py: the ladder report
    stats.py        paired bootstrap; scorers.py: per-record metrics
"""
