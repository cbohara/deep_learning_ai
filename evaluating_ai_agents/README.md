# [Evaluate AI Agents](https://learn.deeplearning.ai/courses/evaluating-ai-agents/lesson/sqkza/introduction?courseName=evaluating-ai-agents)

Course overview
- Create code-based agent that does data analysis
- Agent has access to a
    - Tools that connect to a database to perform analysis
    - Router that identifies which tool to use
    - Memory keeps track of the chat history
- Collect traces of the agent's steps
- Evaluate if the agent chose the right tool based on the query and extracted the right parameters 
- Assess the path takens
- Run structured experiments to iterate on the agent 

Also learn how to monitor agent in production   

## Evaluation intro

Evaluation
- LLM model evaluation - how well LLM performs a specific task
    - MMLU - math, philosophy, medicine, etc
    - Human eval - code generation tasks
- LLM system eval
    - How well the system performs within the broader system/product

Paradigm shift 
- traditional testing 
    - deterministic, like a train on a track
    - write unit tests
    - deterministic results
- LLM testing
    - more like driving a road in a busy city
    - when give the same prompt to an LLM multiple times, will get different results
    - need to examine output quality using coherance 

Common types of eval for LLM systems
- Hallucinations - is the model making stuff up?
- Retrieval relevance
- Does the respond match the user need?
- Toxicity - harmful respones?

Moving from LLM app to agent, adding extra layer of complexity   
Use LLM for reasoning, but then take action on your behalf   

Agent components 
- Reasoning - powered by LLM
- Routing - which tool or skill to use
- Executing - tool, code, etc   

Agent use cases
- Personal assistants
- Automate repetitive tasks
- Research assistants 

Anatomy of an agent 
- Trip planning - Book me a trip to SF
    - Figure out which tool to call - tool selection eval 
    - Search API for available flights + hotels - function call eval
    - Use contex - RAG eval
    - Ask follow up questions to refine 
    - Return friendly + accurate response - tone eval
    - with the correct trip details - correctness eval

Can use HITL or LLMs to evaluate as well   

Even small changes to prompt and code can have unexpected ripple effectives   

Want to iterate on an agents performance   
Cannot rely on deterministic approach   

Tools to evaluate agents  
- Trace instrumentation - understand what the agent is doing under the hood
- Eval runner - contains LLM as a judge  
- Datasets - use to rerun experiemnts 
- Human feedback - note human annotations and instructions

## Agents 
1. Router 
2. Skills
3. Memory and state

User input > router > skills > return result to user   

Router 
- determines which skill to call
- can be
    - LLM equipt with function calling - broad range of capabilities but is a bit more reliable vs rules based code 
    - rule based code - can be more reliable but more limited 

Langgraph + openAI swarm - don't have a single router step, distribute responsibility throughout the agent itself

Skills
- Every agent with have 1+ skills
- Ex: RAG skill - handles embedded input query > vector DB lookup > LLM call with retrieved context 
- Returns result to router 

Memory + state
- Shared state by each component
- Many LLM APIs rely on retrieving each agent step to decide on the next step  
- List of previous steps often stored in memory - like I did with ChatGPT for Slack! 

Example agent - A data analysis assistant that can help you understand sales data from each of your stores  

Skills 
- A data lookup skill to query from your attached database - 3 LLM calls - prepares a database > generate SQL > execute SQL
- A data analysis skill to draw conclusions from your data - 1 LLM call - generate analysis
- A data visualization skill to generate graphs and visualizations about your data - 2 LLM calls - generate chart config + then generate ptyhon code based on config   

User > router (GPT-4o-mini with function calling) > use the tool > return to router > return to user   

# Traces 
Observability
- complete visibility into every layer of the LLM application  
- traces - full run throughs of the app - one end to end run - comprised of multiples spans
- spans - each span is data captured in individual spans of the pipeline 

OpenTelemetry (OTEL)   
Widely used standard for app observability   
Standards on how to capture them   

Mark which code blocks you want to trace, like lambda xray   

Why observability is important 
- Simplifies debugging compared to print statements + logs
- Provides detailed logs across many users 
- Helps understand and sooner control unpredictable behavior of LLMs   

# Techniques for running evals
- Code based evals
    - Most similar to traditional unit tests
    - Run some sort of code on the outputs of the eval
    - Ex: check output matches regex, is JSON parseable, and contains keywords
    - Compare ground truth data 
    - Cosine similarity / cosine distance to do a semantic match
- LLM-as-a-judge
    - Grab the input and ouput of the app
    - Construct a separate prompt 
    - Send the prompt to another LLM
    - Evaluate if response is appropriate or not using discrete classification labels like incorrect vs correct, relevant vs irrelevant, not numeric score 
    - Pros - can run at high scale
    - Cons - need more expensive high end model to evaluate
- Annotations/human labels
    - Humans label responses 
    - Used to train models 
    - Also can get user feedback with thumbs up/down 

When to use each technique?  
- Code based evals - when the output is pretty straightforward to evaluate  
- LLM as a judge is never going to be 100% accurate so non-deterministic but also able to handle vague evaluation  
- Human labels are determinisitic and flexible = ideal, but hard to gather a bunch

Router 
- function calling choice - did the function choose the right one?
- parameter extraction - did the router extract the right function parameters from the question?

Router - can evaluate using LLM as a judge  
Skills - evalute using same tools used to evaluate LLMs since essentially the same thing 

Database lookup tool
- Use LLM as a judge 
- Use code based evaluates of the SQL output

Data analysis 
- LLM as a judge for analysis 

Data viz
- Code based evaluate make sure code runs

# Trajectory evaluations
Say the user asks - Which store had the most sales in 2021?   
use > router (gpt-4o-mini with function calling) > look up sales data tool > data analysis tool > router (gpt-4o-mini with function calling) > user   

If agents in prod have many tools or working in multi-agent system, risk of having a lot of steps = poor efficiency   

Convergence = measures how closely your agent follows the optimal path for a given query   

Test convergence?
1. Run your agent on a set of similar queries
2. Record the number of steps taken
3. Find the length of the optimal path = minimal steps 
4. Calculate convergence store 

Another way to think about convergence score
- What percentage of the time is your agent taking the optimal path, for a given set of inputs?
- Convergence score of 1 means the agent takes the optimal path 100% of the time

Things to keep in mind 
- Convergence evals won't catch situations where an unnecessary step is taken by the agent every time
- Convergence evals should only include complete successful runs, not trajectories that don't go to completion because errored out

## Adding structure to your evaluations
Combine evaluators for end-to-end evaluation workflow   

curate test cases > 
send to different variances (model, prompt, logic) of the agent > 
evaluate the experiment > 
generate score  

LLM apps require iterative performance improvements   

Curate dataset 
- Be more comprehensive than exhaustive - just need 1 or 2 examples of each types of inputs you may get   
- Typically manually generate when getting started and then take examples from production as they are available
- Can provide expected outputs for these test cases 

Track agent changes like
- Prompts 
- Tool definitions
- Router logic
- Underlying model changes

Compontents
- Router - experiment testing different tool description strings - evaluate using code or LLM as a judge 
- Database lookup tool - test different SQL generation prompts to see how they perform by doing code-based eval of SQL generated 
- Database analysis tool - test using different LLM models - use LLM as a judge eval to see how it performs 

Why add LLM as a judge evaluator in addition to code eval?   
Can be more scalable vs code eval    
Can use the same techniques used to judge the agent as you can use to judge the judge :mindblow:    
Can use semantic similarility as an alternative to direct string comparison    

## Monitoring agents in prod 

4 steps from dev to prod
1. Choose the right agent architecture
2. Choose what to evaluate and the correct metrics
3. Build eval structure (end-to-end eval)
4. Use your data to iterate 

Multi-agent system = agents call other agents :mindblow:   
Multi-modal system = single agent can handle different inputs like audio/video/unstructured data/etc   

What's different about prod?
- New failure modes - users provide unexpected inputs > failures
- Higher complexity when calling to external APIs and other agents

Tools for managing agent performance in prod 
- Instrumentation and feedback
- Monitor your metrics
- CI/CD 

If eval metrics don't agree with collected user feedback from app, need to adapt   

Monitor metrics
- Efficiency of system (convergence eval)
- Dependencies on external services (latency, cost)
    - Ex: different model providers, different model types can have big impact 

Want to manage data sets + continuously add to them   

Can even build a self-improving agent!   

## 📚 Resources

- In addition to assessing the convergence of your agent's path, you can also analyze the failure probabilities of each step of your agent. Here's a [reference example](https://blog.langchain.dev/scipe-systematic-chain-improvement-and-problem-evaluation/) on how this can be done.
-  [Tracing and Evaluating AI Agents built with LlamaIndex's Workflows](https://arize.com/resource/llamaindex-workflows-everything-you-need-to-get-started-and-trace-and-evaluate-your-agent/)
-  Evaluating a LangChain tool-calling agent: [Link to notebook](https://github.com/Arize-ai/phoenix/blob/b107d9bc848efd38f030a8c72954e89616c43723/tutorials/evals/evaluate_tool_calling.ipynb), [Link to Video](https://www.youtube.com/watch?v=EfhylWtNb1s&t=254s)
-  [Tracing and Evaluating LangGraph Agents](https://arize.com/blog/langgraph/)
-  [Prompt template iteration for a summarization Service](https://github.com/Arize-ai/phoenix/blob/main/tutorials/experiments/summarization.ipynb)
-  Comparing Agent Frameworks: [Link to reading](https://arize.com/blog-course/llm-agent-how-to-set-up/comparing-agent-frameworks/), [Link to repo](https://github.com/Arize-ai/phoenix/tree/main/examples/agent_framework_comparison)
-  [Discussion of multi-agent framework with AutoGen](https://www.youtube.com/watch?v=fuvcV8o5wb0&list=PL86ARIu_ElO7HOm7cVgzfEs_NwSdfhHFA&index=6&ab_channel=ArizeAI)
-  [Arize Phoenix Documentation](https://docs.arize.com/phoenix)