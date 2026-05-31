import argparse
import sys
from agents.graph import create_gis_graph
from core.state import AgentState

def main():
    parser = argparse.ArgumentParser(description="GeoGraph CLI - GIS Multi-Agent Collaboration System")
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        help="The spatial GIS problem or programming request to solve."
    )
    parser.add_argument(
        "--provider", 
        type=str, 
        default=None, 
        help="LLM provider (openai, deepseek, gemini, anthropic, ollama)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Specific model name to use."
    )
    parser.add_argument(
        "--temp", 
        type=float, 
        default=None, 
        help="LLM temperature (0.0 to 1.0)."
    )
    parser.add_argument(
        "--max-iter", 
        type=int, 
        default=3, 
        help="Maximum loops between GIS Developer and GIS QA."
    )

    args = parser.parse_args()

    # Compile the graph
    print("Compiling GeoGraph Workflow Graph...")
    graph = create_gis_graph()

    # Initialize state
    initial_state = AgentState(
        task=args.task,
        plan="",
        draft_code="",
        explanation="",
        qa_feedback="",
        qa_approved=False,
        iterations=0,
        max_iterations=args.max_iter,
        logs=[]
    )

    # Configure runnable settings
    config = {
        "configurable": {
            "provider": args.provider,
            "model_name": args.model,
            "temperature": args.temp
        }
    }

    print(f"\nStarting Collaboration workflow for GIS Task:\n\"{args.task}\"\n")
    print("=" * 60)

    # Run the graph and stream events/updates
    # We can stream state updates or run it in one go.
    try:
        final_state = graph.invoke(initial_state, config=config)
        
        print("\n" + "=" * 60)
        print("GeoGraph Workflow Execution Timeline:")
        print("=" * 60)
        
        # Display the log history
        for entry in final_state.get("logs", []):
            agent = entry.get("agent")
            log_type = entry.get("log_type").upper()
            content = entry.get("content")
            time = entry.get("timestamp")
            
            print(f"\n[{time}] {agent} ({log_type}):")
            print("-" * 40)
            print(content)
            
        print("\n" + "=" * 60)
        if final_state.get("qa_approved", False):
            print("Workflow Status: SUCCESS - QA APPROVED")
            print("The final GIS script has been saved to: C:\\Users\\26503\\Documents\\antigravity\\magical-hertz\\gis_analysis.py")
        else:
            print("Workflow Status: FINISHED (Not approved by QA or reached max iterations)")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running workflow graph: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
