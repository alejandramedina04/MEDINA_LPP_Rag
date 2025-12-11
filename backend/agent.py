# =============================================================================
# Agent module for RAG Assistant
# =============================================================================
# This file creates an AI agent that can DECIDE when to search the database.
# Instead of always retrieving passages, the LLM chooses when retrieval helps.
# =============================================================================

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from backend.database import RAGDatabase

class RAGAgent:
    def __init__(self, db: RAGDatabase, model_name: str, max_iter: int):
        self.db = db
        self.model_name = model_name
        self.max_iter = max_iter
        self.last_sources = []  # We'll store retrieved passages here for the UI

    def create_tool(self):
        # ---------------------------------------------------------------------
        # The @tool decorator transforms this function into something the
        # LLM can call. The docstring is CRUCIAL—it's what the LLM reads
        # to decide whether and how to use this tool.
        # ---------------------------------------------------------------------
        @tool("Query RAG Database")
        def query_rag_db(query: str) -> str:
            """Search the vector database containing curated sources about artificial intelligence.
            
            Args:
                query: Search query about artificial intelligence (its uses, benefits, risks, ethics, and black-box issues).
                
            Returns:
                Relevant passages from the database for the model to use when answering. 
            """
            try:
                results = self.db.query(query)
                
                if results:
                    # Store sources for UI display
                    self.last_sources.extend(results)
                    
                    # Format passages for the LLM to read
                    passages = [row["text"] for row in results]
                    return "\n\n---\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)])
                else:
                    return "No relevant passages found."
                    
            except Exception as e:
                return f"Error querying database: {str(e)}"
        
        return query_rag_db

    # TO DO: Update the ask() function
    def ask(self, question: str) -> dict:
        """
        Ask a question to the agent.
        
        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Reset sources for this query
        self.last_sources = []
        
        # TO DO: Create the LLM instance
        llm = LLM(model = self.model_name)

        # TO DO: Call the database tool (e.g. the function above)
        query_tool = self.create_tool()
        

        agent = Agent(
            role='Artificial Intelligence Content Assistant',
            goal='Answer questions about artificial intelligence using the vector database.',
            backstory=(
                "You are an expert on artificial intelligence with access to a database of "
                "curated sources about AI's uses, benefits, risks, black-box behavior, and ethics. "
                "Base your answers only on the retrieved passages, and if the information "
                "is not present, say that it is not covered in the sources."),
            tools=[query_tool],
            llm=llm,
            verbose=True, # Shows what the agent is doing
            allow_delegation=False, # Does not create sub-agents
            max_iter=self.max_iter # Limits tool calls
        )
        
        # TO DO: Create the task
        task = Task(
            description = (
                f"Answer this question about artificial intelligence using ONLY the retrieved passages from the RAG database:\n\n"
                f"{question}"
            ),
            agent = agent,
            expected_output = (
                "Write the answer in a string format with this structure:\n\n"
                "Short answer:\n"
                "- 1-3 sentences that directly and simply answer the question.\n\n"
                "Details:\n"
                "- 3-6 bullet points explaining the key ideas in a clear, friendly way.\n\n"
                "From the sources:\n"
                "- 2-4 bullet points that say what comes from the retrieved passages (e.g., 'The sources say...').\n\n"
                "Next steps or tips:\n"
                "- 1-3 bullet points suggesting what the user could read, check, or think about next."
            )
        )

        # TO DO: Create the Crew and run it
        crew = Crew(agents = [agent],
                    tasks = [task],
                    verbose = True,
                    max_rpm = 20)
        
        result = crew.kickoff()
        
        # Returns the answer and sources
        return {
            "answer": str(result),
            "sources": self.last_sources.copy()
        }

