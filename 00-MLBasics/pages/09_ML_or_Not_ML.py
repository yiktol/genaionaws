import streamlit as st
import random


def next_question():
    st.session_state.questions_asked += 1
    # st.rerun()

def main():
    st.title("Machine Learning Decision Game")
    st.subheader("Can you decide when to use Machine Learning?")
    
    # Introduction
    st.markdown("""
    In this game, you'll be presented with various business scenarios.
    Your task is to decide whether machine learning would be an appropriate solution
    for each scenario based on the principles of when to use ML:
    
    - Use ML when you can't code it (complex tasks where deterministic solutions don't suffice)
    - Use ML when you can't scale it (replace repetitive tasks needing human-like expertise)
    - Use ML when you have to adapt/personalize
    - Use ML when you can't track it
    
    Test your knowledge and see if you can make the right decisions!
    """)
    
    # Game state
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = 0
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    
    # Scenarios data - tuples of (scenario, should_use_ml, explanation)
    scenarios = [
        ("A bank needs to calculate compound interest for customer accounts.", 
         False, 
         "This is a deterministic calculation with clear mathematical formulas. Traditional programming is more appropriate."),
        
        ("An e-commerce website wants to recommend products based on customer browsing history.", 
         True, 
         "Personalization at scale is a perfect ML use case. The patterns are complex and need to adapt to individual users."),
        
        ("A healthcare provider needs to analyze medical images to detect abnormalities.", 
         True, 
         "Image recognition for medical diagnosis is complex and benefits greatly from ML, which can detect patterns humans might miss."),
        
        ("A car manufacturing company wants to automate quality control by detecting defects in parts.", 
         True, 
         "Visual inspection at scale involves complex pattern recognition that's ideal for ML."),
        
        ("A financial institution needs to detect potentially fraudulent transactions in real-time.", 
         True, 
         "Fraud detection involves complex patterns that change over time, making it perfect for ML."),
        
        ("A utility company needs to calculate monthly bills based on meter readings.", 
         False, 
         "This is a straightforward calculation with clear rules that can be handled by traditional programming."),
        
        ("A streaming service wants to suggest content based on what similar users have enjoyed.", 
         True, 
         "Content recommendation systems benefit from ML to identify complex patterns and personalize at scale."),
        
        ("A payroll system needs to calculate employee taxes based on current tax laws.", 
         False, 
         "Tax calculations follow explicit rules and formulas, making traditional programming more appropriate."),
        
        ("A social media platform wants to automatically moderate content to identify harmful posts.", 
         True, 
         "Content moderation involves complex language understanding and contextual awareness, ideal for ML."),
        
        ("A logistics company wants to optimize delivery routes across a city.", 
         True, 
         "Route optimization with multiple variables and constraints can benefit from ML, especially reinforcement learning."),
        
        ("An inventory system needs to track product counts and generate alerts when stock is low.", 
         False, 
         "Simple threshold-based alerting can be handled with traditional programming rules."),
        
        ("A customer support system needs to categorize incoming support tickets by department.", 
         True, 
         "Text classification for routing tickets benefits from ML, especially as the categories and language evolve."),
        
        ("A smart home device needs to understand and respond to voice commands.", 
         True, 
         "Speech recognition is a complex problem that benefits greatly from ML."),
        
        ("An HR system needs to track employee vacation days and enforce company policies.", 
         False, 
         "This involves clear business rules that can be coded directly without ML."),
        
        ("A vehicle needs to navigate autonomously in unpredictable real-world environments.", 
         True, 
         "Autonomous driving involves complex perception and decision-making that's ideal for ML and reinforcement learning.")
    ]
    
    # Start/Restart button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.game_active:
            if st.button("Start Game", use_container_width=True, type="primary"):
                st.session_state.game_active = True
                st.session_state.score = 0
                st.session_state.questions_asked = 0
                st.session_state.scenarios = random.sample(scenarios, len(scenarios))  # Shuffle scenarios
                st.rerun()
    
    # Game logic
    if st.session_state.game_active:
        # Display progress
        st.progress(st.session_state.questions_asked / 10)
        st.write(f"Question: {st.session_state.questions_asked + 1}/10")
        st.write(f"Score: {st.session_state.score}")
        
        # Get current scenario
        if st.session_state.questions_asked < 10:
            current_scenario = st.session_state.scenarios[st.session_state.questions_asked]
            
            # Display scenario
            st.markdown(f"### Scenario:")
            st.markdown(f"**{current_scenario[0]}**")
            
            # Get user choice
            col1, col2 = st.columns(2)
            with col1:
                use_ml = st.button("Use Machine Learning", use_container_width=True)
            with col2:
                dont_use_ml = st.button("Don't Use Machine Learning", use_container_width=True)
            
            # Process answer
            if use_ml or dont_use_ml:
                user_choice = use_ml  # True if they chose ML, False if they chose not to use ML
                correct = user_choice == current_scenario[1]
                
                if correct:
                    st.success("Correct! 🎉")
                    st.session_state.score += 1
                else:
                    st.error("Incorrect! 😕")
                
                # Show explanation
                st.info(f"Explanation: {current_scenario[2]}")
                
                # Move the Next Question button outside the if block
                st.button("Next Question", on_click=lambda: next_question())
                
                # Next question button
                # if st.button("Next Question"):
                #     st.session_state.questions_asked += 1
                #     st.rerun()
        else:
            # Game over
            final_score = st.session_state.score
            
            st.markdown(f"## Game Over!")
            st.markdown(f"### Your final score: {final_score}/10")
            
            # Provide feedback based on score
            if final_score == 10:
                st.balloons()
                st.success("Perfect score! You're an ML decision-making expert!")
            elif final_score >= 8:
                st.success("Great job! You have a strong understanding of when to apply ML!")
            elif final_score >= 6:
                st.info("Good effort! You understand the basics but might want to review some concepts.")
            else:
                st.warning("You might want to review the key principles of when to use ML vs traditional programming.")
            
            # Key takeaways
            st.markdown("""
            ### Key Takeaways:
            
            Remember to use Machine Learning when:
            - Tasks are too complex for explicit programming (image recognition, natural language understanding)
            - You need to scale human-like expertise (recommendations, content moderation)
            - Systems need to adapt and personalize (user preferences, dynamic environments)
            - Problems involve unpredictable environments or complex patterns (autonomous vehicles, fraud detection)
            
            Traditional programming is better when:
            - Problems have clear, unchanging rules (calculations, rule-based workflows)
            - Transparency and auditability are critical (financial calculations, compliance)
            - You have limited data
            - The solution requires perfect accuracy
            """)
            
            if st.button("Play Again"):
                st.session_state.game_active = False
                st.rerun()

if __name__ == "__main__":
    main()