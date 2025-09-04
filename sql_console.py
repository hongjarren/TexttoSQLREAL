from texttosql import TextToSQLModel
import os
import datetime
import json
import glob

def get_latest_model():
    """Find the most recently created model directory"""
    model_dirs = []

    base_paths = [
        ("./text_to_sql_results/final_model", 1),  # Non-timestamped directory
        ("./text_to_sql_results_*/final_model", 0),  # Timestamped directories
        ("./text_to_sql_model", -1)  # Standard directory
    ]

    # Collect all available model directories
    for pattern, base_priority in base_paths:
        if "*" in pattern:
            # Handle wildcard patterns
            for dir_path in glob.glob(pattern):
                if os.path.exists(dir_path):
                    mtime = os.path.getmtime(dir_path)
                    model_dirs.append({
                        "path": dir_path,
                        "time": mtime,
                        "priority": base_priority
                    })
        else:
            # Handle direct paths
            if os.path.exists(pattern):
                mtime = os.path.getmtime(pattern)
                model_dirs.append({
                    "path": pattern,
                    "time": mtime,
                    "priority": base_priority
                })

    if not model_dirs:
        return None

    # First, find the most recent modification time
    latest_time = max(d["time"] for d in model_dirs)
    
    # Consider a model "current" if it's within 5 seconds of the latest modification
    RECENCY_THRESHOLD = 5  # seconds
    current_models = [d for d in model_dirs 
                     if latest_time - d["time"] <= RECENCY_THRESHOLD]
    
    # If we have multiple "current" models, use priority as tiebreaker
    # Otherwise, just use the most recent one
    if current_models:
        return sorted(current_models, 
                     key=lambda x: (x["time"], x["priority"]), 
                     reverse=True)[0]["path"]
    
    # Fallback to most recent if no models are "current"
    return sorted(model_dirs, 
                 key=lambda x: x["time"], 
                 reverse=True)[0]["path"]

def interactive_mode():
    # Initialize the model
    text_to_sql = TextToSQLModel()
    
    # Check if model exists and load it
    model_path = get_latest_model()
    if not model_path:
        print("Error: Trained model not found. Please ensure the model is saved at './text_to_sql_results/final_model' or './text_to_sql_model'")
        return
    
    print(f"Loading model from {model_path}...")
    text_to_sql.load_model(model_path)
    print("Model loaded successfully!")
    
    print("\n=== Text-to-SQL Converter ===")
    print("Type your questions in natural language, and I'll convert them to SQL.")
    print("Type 'exit' to quit, 'help' for examples, 'stats' for performance stats, 'retrain' to retrain the model, 'cleanup' to remove old models.")
    
    while True:
        print("\n" + "-" * 50)
        question = input("Enter your question: ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
            
        if question.lower() == 'help':
            show_examples()
            continue
            
        if question.lower() == 'stats':
            show_stats(text_to_sql)
            continue
            
        if question.lower() == 'retrain':
            retrain_model(text_to_sql)
            continue
            
        if question.lower() == 'cleanup':
            cleanup_old_models()
            continue
            
        if question.lower() == 'sync':
            # Force a sync with the latest model
            latest_model = get_latest_model()
            if latest_model:
                print(f"Syncing with latest model: {latest_model}")
                text_to_sql.load_model(latest_model)
                print("Model synced successfully!")
            else:
                print("No models found to sync with.")
            continue
            
        if not question.strip():
            print("Please enter a question.")
            continue
        
        print("Generating SQL...")
        sql_query = text_to_sql.generate_sql(question)
        
        print("\nGenerated SQL:")
        print("-" * 50)
        print(sql_query)
        print("-" * 50)
        
        # Ask if the SQL is correct
        feedback = input("\nIs this SQL correct? (y/n): ")
        if feedback.lower() == 'n':
            print("Please provide the correct SQL query:")
            corrected_sql = input("> ")
            
            if corrected_sql.strip():
                # Add the correction to the feedback data
                should_retrain = text_to_sql.add_correction(question, sql_query, corrected_sql)
                print("Thanks for the correction! This will help improve future results.")
                
                if should_retrain:
                    print("\nWe have collected several corrections. Would you like to retrain the model now? (y/n)")
                    if input("> ").lower() == 'y':
                        retrain_model(text_to_sql)
            else:
                print("No correction provided. Continuing without feedback.")
        else:
            # If the SQL is correct, do not add to feedback
            print("Great! Thanks for the feedback. No changes will be made to the feedback data.")

def show_examples():
    examples = [
        "Give me all part numbers created by ER code ABC123",
        "Show me all parts created after January 2023",
        "Count how many parts each engineer has created",
        "Find parts with quantity less than 10",
        "List all parts and their suppliers"
    ]
    
    print("\nExample questions you can ask:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

def show_stats(text_to_sql):
    """Display performance statistics"""
    stats = text_to_sql.analyze_performance()
    
    print("\n=== Performance Statistics ===")
    print(f"Total queries: {stats['metrics']['total_queries']}")
    print(f"Pattern matched: {stats['metrics']['pattern_matched']}")
    print(f"Model generated: {stats['metrics']['model_generated']}")
    print(f"User corrected: {stats['metrics']['user_corrected']}")
    print(f"Success rate: {stats['metrics']['success_rate']*100:.1f}%")
    
    print("\n=== Feedback Statistics ===")
    print(f"Total corrections: {stats['feedback_stats']['total_corrections']}")
    print(f"Total examples: {stats['feedback_stats']['total_examples']}")
    
    if stats['feedback_stats']['last_retrained']:
        print(f"Last retrained: {stats['feedback_stats']['last_retrained']}")
    else:
        print("Model has not been retrained yet")
    
    if stats['error_patterns']:
        print("\n=== Common Error Patterns ===")
        for pattern, count in stats['error_patterns'].items():
            print(f"{pattern}: {count} occurrences")

def retrain_model(text_to_sql):
    """Retrain the model with feedback data"""
    print("\nRetraining model with feedback data...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./text_to_sql_results_{timestamp}"
    
    success = text_to_sql.retrain_from_feedback(output_dir=output_dir)
    
    if success:
        # Update the last_retrained timestamp in feedback data
        text_to_sql.feedback_data["metadata"]["last_retrained"] = datetime.datetime.now().isoformat()
        text_to_sql._save_feedback_data()
        
        print(f"Model successfully retrained and saved to {output_dir}/final_model")
        
        # Evaluate the model on the test dataset
        eval_results = text_to_sql.evaluate_model(text_to_sql.load_data()["test"])
        print(f"Model accuracy after retraining: {eval_results['accuracy']:.2f}")
        
        print("Would you like to load the new model now? (y/n)")
        if input("> ").lower() == 'y':
            text_to_sql.load_model(os.path.join(output_dir, "final_model"))
            print("New model loaded successfully!")
    else:
        print("Retraining failed or no feedback data available.")

def cleanup_old_models():
    """Delete old model directories to save space"""
    print("\nCleaning up old model directories...")
    
    # Get all directories that match the pattern
    model_dirs = [d for d in os.listdir("./") if d.startswith("text_to_sql_results_")]
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    
    # Keep the most recent one
    if model_dirs and len(model_dirs) > 1:
        print(f"Keeping most recent model: {model_dirs[0]}")
        for old_dir in model_dirs[1:]:
            print(f"Removing old model directory: {old_dir}")
            try:
                import shutil
                shutil.rmtree(old_dir)
            except Exception as e:
                print(f"Error removing directory {old_dir}: {e}")
    else:
        print("No model directories found to clean up.")

if __name__ == "__main__":
    interactive_mode()



