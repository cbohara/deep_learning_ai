# [Claude Code](https://learn.deeplearning.ai/courses/claude-code-a-highly-agentic-coding-assistant/lesson/66b35/introduction)

Interact with CLI -> agentic coding assistant   

Assistant contains
- claude models - opus, sonnet
- memory
- tools

Model will
- gather context ->
- formulate a plan ->
- take an action -> loop

Can be used to 
- discover
    - explore codebase + history
    - search docs
- design
    - plan a project
    - develop tech specs
    - define architecture
- build
    - implement code
    - write + execute tests
    - create commits + PRs
- deploy
    - automate CI/CD
    - configure envs
    - manage deployments
- support + scale
    - debug errors
    - monitor usage + performance

model is given plain text directions ->   
when model responds with a request to use a tool ->   
coding assistant does whatever the tool is supposed to do   

tools include 
- bash - run a shell command
- edit - edit a file
- read - read a file

tool use strength enables claude code to gather context needed
- combine diff tools to handle complex work
- easily navigate codebase 
- can add additional tools to claude code by connecting MCP servers 

`claude.md`
- how memory is managed across sessions
- define style guide + common commands
- auto loaded into context 
- convo stored locally on machine

