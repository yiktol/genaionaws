# Machine Learning Types Matcher Game

import streamlit as st
import random
import pandas as pd
import time

def main():
    st.title("Machine Learning Types Matcher")
    st.subheader("Can you identify the correct machine learning approach for each scenario?")
    
    # Introduction
    st.markdown("""
    In this game, you'll be presented with various scenarios where machine learning can be applied.
    Your task is to identify which type of machine learning would be most appropriate:
    
    - **Supervised Learning**: Training with labeled data to make predictions or classifications
    - **Unsupervised Learning**: Finding patterns in unlabeled data
    - **Reinforcement Learning**: Learning through trial and error with rewards and penalties
    - **Self-Supervised Learning**: Creating labels from the data itself for pretraining
    
    Test your knowledge of machine learning fundamentals!
    """)
    
    # Game state
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = 0
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []
        
    # ML Type scenarios - (scenario, correct_type, explanation)
    scenarios = [
        (
            "A company wants to predict future sales based on historical sales data where each data point includes the date, marketing spend, and resulting sales numbers.",
            "Supervised Learning",
            "This is a supervised learning problem because we have labeled data (historical sales with known outcomes) and need to predict a specific target value (future sales)."
        ),
        (
            "An e-commerce website wants to group their customers into distinct segments based on purchasing behaviors without any predefined categories.",
            "Unsupervised Learning",
            "This is an unsupervised learning problem (specifically clustering) because we're looking for patterns and segments in the data without having predefined labels."
        ),
        (
            "A company is developing an AI system for a self-driving car that needs to learn optimal driving behaviors through interactions with a simulated environment.",
            "Reinforcement Learning",
            "This is a reinforcement learning problem because the system learns by interacting with an environment (driving conditions), receiving feedback (rewards for safe driving, penalties for accidents), and adjusting behavior accordingly."
        ),
        (
            "A research team wants to train a language model by having it predict the next word in sentences from a large corpus of text.",
            "Self-Supervised Learning",
            "This is a self-supervised learning problem because the model creates its own supervision signal from the data (predicting masked or future words) without requiring external labels."
        ),
        (
            "A medical team has thousands of labeled X-ray images categorized as 'pneumonia' or 'no pneumonia' and wants to build a model to classify new X-rays.",
            "Supervised Learning",
            "This is a supervised learning problem (specifically binary classification) because we have labeled training data (X-rays with known diagnoses) and need to classify new images into one of two categories."
        ),
        (
            "A data scientist wants to detect unusual patterns in network traffic that might indicate security breaches without knowing in advance what those patterns look like.",
            "Unsupervised Learning",
            "This is an unsupervised learning problem (specifically anomaly detection) because we're looking for unusual patterns without having labeled examples of what constitutes 'normal' vs 'abnormal'."
        ),
        (
            "A large language model is being trained by masking random words in sentences and having the model predict what those words should be.",
            "Self-Supervised Learning",
            "This is self-supervised learning because the task creates its own labels from the input data by masking words and using the original unmasked text as the supervision signal."
        ),
        (
            "A robotics engineer is developing an algorithm to teach a robot arm to pick up and sort objects of different shapes and weights through trial and error.",
            "Reinforcement Learning",
            "This is a reinforcement learning problem because the robot learns optimal behaviors through trial and error, receiving rewards for successful grasps and penalties for drops or mistakes."
        ),
        (
            "A financial services company has customer transaction histories labeled as 'fraudulent' or 'legitimate' and wants to build a model to detect fraud in new transactions.",
            "Supervised Learning",
            "This is a supervised learning classification problem because we have labeled examples of fraudulent and legitimate transactions to train a model that can classify new transactions."
        ),
        (
            "A streaming service wants to group movies into genres based on their content, dialogue, and visual style without using predefined genre categories.",
            "Unsupervised Learning",
            "This is an unsupervised learning clustering problem because we're looking to identify natural groupings in the data without predefined labels."
        ),
        (
            "An AI researcher is developing a system where an agent learns to play chess by playing against itself and improving based on game outcomes.",
            "Reinforcement Learning",
            "This is a reinforcement learning problem because the agent learns optimal strategies through trial and error gameplay, receiving rewards for winning and penalties for losing."
        ),
        (
            "A company wants to build a recommendation system based on past user ratings of products to predict what ratings users would give to products they haven't seen yet.",
            "Supervised Learning",
            "This is a supervised learning problem because we have labeled data (known user ratings) that we use to predict unknown values (ratings for unseen products)."
        ),
        (
            "A video platform is training an AI system by having it predict what happens next in video sequences.",
            "Self-Supervised Learning",
            "This is self-supervised learning because the model creates its own supervision signal from the data (using earlier frames to predict later frames) without requiring external labels."
        ),
        (
            "A data analyst has a large dataset of customer purchase records and wants to discover underlying patterns without looking for anything specific.",
            "Unsupervised Learning",
            "This is an unsupervised learning problem because we're exploring data to find patterns and structures without having labeled examples or specific target variables."
        ),
        (
            "A conversational AI is being trained by predicting masked portions of dialogue based on surrounding context from millions of conversations.",
            "Self-Supervised Learning",
            "This is self-supervised learning because the model creates supervision signals from the raw data itself by predicting masked content based on context."
        ),
        (
            "A gaming company is developing an AI that learns to play a new video game by maximizing its score through repeated gameplay attempts.",
            "Reinforcement Learning",
            "This is a reinforcement learning problem because the AI learns optimal strategies through trial and error, receiving rewards (points in the game) and adjusting behavior accordingly."
        ),
        (
            "An image analysis tool is being trained by having it predict the original color of black and white images.",
            "Self-Supervised Learning",
            "This is self-supervised learning because the model creates its own supervision (the original color) from the input data (grayscale version) without external labeling."
        ),
        (
            "A grocery chain wants to predict daily sales for each product category based on historical sales data that includes day of week, promotions, and holidays.",
            "Supervised Learning",
            "This is a supervised learning problem (specifically regression) because we have labeled historical data (past sales figures) and need to predict a numerical target value."
        ),
        (
            "A research team wants to identify potential new drug compounds by having an AI generate molecular structures that maximize certain biological properties.",
            "Reinforcement Learning",
            "This is a reinforcement learning problem because the model learns to generate structures that maximize a reward function (biological effectiveness) through an iterative process of generation and evaluation."
        ),
        (
            "A social network wants to analyze user connections to identify communities of closely connected individuals without predefined groupings.",
            "Unsupervised Learning",
            "This is an unsupervised learning problem (specifically community detection) because we're looking for natural groupings in network data without labeled examples."
        )
    ]
    
    # Start/Restart button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.game_active:
            if st.button("Start Game", use_container_width=True, type="primary"):
                st.session_state.game_active = True
                st.session_state.score = 0
                st.session_state.questions_asked = 0
                st.session_state.scenarios = random.sample(scenarios, 10)  # Pick 10 random scenarios
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
            scenario_text = current_scenario[0]
            correct_answer = current_scenario[1]
            explanation = current_scenario[2]
            
            # ML types options
            ml_types = ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Self-Supervised Learning"]
            
            # Display scenario
            st.markdown("### Scenario:")
            st.markdown(f"**{scenario_text}**")
            
            # Display options
            st.markdown("#### What type of machine learning would be most appropriate for this scenario?")
            
            # Use radio button for ML type selection
            user_answer = st.radio("Select the best approach:", ml_types, key=f"q{st.session_state.questions_asked}")

            # Create columns for the buttons
            col1, col2 = st.columns(2)
            
            # Submit button in first column
            with col1:
                if st.button("Submit Answer", key=f"submit{st.session_state.questions_asked}"):
                    if user_answer == correct_answer:
                        st.success("✅ Correct! That's the right machine learning type for this scenario.")
                        st.session_state.score += 1
                    else:
                        st.error(f"❌ Incorrect. The right approach is {correct_answer}.")
                    
                    # Show explanation
                    st.info(f"**Explanation**: {explanation}")
                    st.session_state.answer_submitted = True

            # Next question button in second column
            with col2:
                if getattr(st.session_state, 'answer_submitted', False):
                    if st.button("Next Question", key=f"next{st.session_state.questions_asked}"):
                        st.session_state.questions_asked += 1
                        st.session_state.answer_submitted = False
                        st.rerun()
            
            # # Check answer button
            # if st.button("Submit Answer", key=f"submit{st.session_state.questions_asked}"):
            #     if user_answer == correct_answer:
            #         st.success("✅ Correct! That's the right machine learning type for this scenario.")
            #         st.session_state.score += 1
            #     else:
            #         st.error(f"❌ Incorrect. The right approach is {correct_answer}.")
                
            #     # Show explanation
            #     st.info(f"**Explanation**: {explanation}")
                
            #     # Next question button
            #     if st.button("Next Question", key=f"next{st.session_state.questions_asked}"):
            #         st.session_state.questions_asked += 1
            #         st.rerun()
        else:
            # Game over
            final_score = st.session_state.score
            
            st.markdown(f"## Game Over!")
            st.markdown(f"### Your final score: {final_score}/10")
            
            # Provide feedback based on score
            if final_score == 10:
                st.balloons()
                st.success("Perfect score! You're a machine learning expert!")
            elif final_score >= 8:
                st.success("Great job! You have a strong understanding of machine learning types!")
            elif final_score >= 6:
                st.info("Good effort! You understand the basics but might want to review some concepts.")
            else:
                st.warning("You might want to review the different types of machine learning approaches.")
            
            # Display a summary table
            st.markdown("### Machine Learning Types Summary")
            
            ml_types_data = [
                {"Type": "Supervised Learning", "Description": "Learning from labeled data to make predictions or classifications", "Examples": "Image classification, speech recognition, regression problems, spam detection"},
                {"Type": "Unsupervised Learning", "Description": "Finding patterns or structures in unlabeled data", "Examples": "Clustering, anomaly detection, dimensionality reduction, recommendation systems"},
                {"Type": "Reinforcement Learning", "Description": "Learning optimal behaviors through trial and error with rewards and penalties", "Examples": "Game playing, robotics, autonomous driving, resource management"},
                {"Type": "Self-Supervised Learning", "Description": "Creating supervision signals from the data itself for pretraining", "Examples": "Language models predicting masked words, contrastive learning, pretext tasks"},
            ]
            
            df = pd.DataFrame(ml_types_data)
            st.table(df)
            
            # Key characteristics
            st.markdown("### Key Characteristics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Supervised Learning:**")
                st.markdown("- Requires labeled training data")
                st.markdown("- Has a clear target variable to predict")
                st.markdown("- Includes classification and regression")
                st.markdown("- Evaluation is straightforward with test data")
                
                st.markdown("**Unsupervised Learning:**")
                st.markdown("- Uses unlabeled data")
                st.markdown("- Finds hidden patterns or structures")
                st.markdown("- Includes clustering and dimensionality reduction")
                st.markdown("- Evaluation can be more subjective")
            
            with col2:
                st.markdown("**Reinforcement Learning:**")
                st.markdown("- Agent interacts with environment")
                st.markdown("- Learns through trial and error")
                st.markdown("- Receives rewards and penalties")
                st.markdown("- Optimizes for long-term reward")
                
                st.markdown("**Self-Supervised Learning:**")
                st.markdown("- Creates its own supervision from data")
                st.markdown("- Often used for pretraining large models")
                st.markdown("- Can learn from massive unlabeled datasets")
                st.markdown("- Includes techniques like masked language modeling")
            
            if st.button("Play Again"):
                st.session_state.game_active = False
                st.rerun()

if __name__ == "__main__":
    main()
