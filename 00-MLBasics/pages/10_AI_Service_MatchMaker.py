# AWS AI Service Matchmaker Game

import streamlit as st
import random
import pandas as pd

def main():
    st.title("AWS AI Service Matchmaker")
    st.subheader("Can you match the scenario to the correct AWS AI service?")
    
    # Introduction
    st.markdown("""
    In this game, you'll be presented with various business scenarios where AWS AI services can provide solutions.
    Your task is to select the most appropriate AWS AI service for each scenario.
    
    Test your knowledge of AWS AI capabilities and see how well you can match services to use cases!
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
        
    # AWS AI Services scenarios - (scenario, correct_service, explanation, incorrect_options)
    scenarios = [
        (
            "A retail company wants to provide personalized product recommendations to customers based on their browsing history and purchase patterns.",
            "Amazon Personalize",
            "Amazon Personalize provides real-time personalization and recommendations using the same technology used by Amazon.com. It's ideal for product recommendations based on user behavior.",
            ["Amazon Rekognition", "Amazon Comprehend", "Amazon Kendra"]
        ),
        (
            "A media company needs to automatically identify celebrities in their video content to improve searchability.",
            "Amazon Rekognition",
            "Amazon Rekognition can analyze images and videos to identify objects, people, text, scenes, and activities, including celebrity recognition.",
            ["Amazon Textract", "Amazon Comprehend", "Amazon Kendra"]
        ),
        (
            "A financial institution wants to automatically extract information from scanned loan application documents.",
            "Amazon Textract",
            "Amazon Textract is designed to extract text and data from scanned documents. It goes beyond simple OCR to identify form fields and tables.",
            ["Amazon Rekognition", "Amazon Comprehend", "Amazon Transcribe"]
        ),
        (
            "A company wants to implement a search function for their internal knowledge base that can understand natural language queries.",
            "Amazon Kendra",
            "Amazon Kendra is an intelligent search service that uses natural language processing to return specific answers to questions, making it ideal for knowledge bases.",
            ["Amazon Comprehend", "Amazon Lex", "Amazon Personalize"]
        ),
        (
            "A healthcare provider wants to analyze patient records to identify key medical information and classify documents.",
            "Amazon Comprehend Medical",
            "Amazon Comprehend Medical is specifically designed to extract information from unstructured medical text using NLP.",
            ["Amazon Rekognition", "Amazon Textract", "Amazon Kendra"]
        ),
        (
            "A streaming service wants to automatically generate subtitles for their video content in multiple languages.",
            "Amazon Transcribe",
            "Amazon Transcribe automatically converts speech to text and can be combined with Amazon Translate for multilingual subtitling.",
            ["Amazon Polly", "Amazon Rekognition", "Amazon Textract"]
        ),
        (
            "An e-commerce company wants to detect potentially fraudulent activities in their online transactions.",
            "Amazon Fraud Detector",
            "Amazon Fraud Detector is specifically designed to identify potentially fraudulent online activities using machine learning.",
            ["Amazon Comprehend", "Amazon Macie", "Amazon Rekognition"]
        ),
        (
            "A company wants to build a chatbot that can communicate with customers and handle basic service requests.",
            "Amazon Lex",
            "Amazon Lex provides the advanced deep learning capabilities of automatic speech recognition (ASR) and natural language understanding (NLU) to build conversational interfaces like chatbots.",
            ["Amazon Polly", "Amazon Comprehend", "Amazon Connect"]
        ),
        (
            "A company needs to create realistic voice narrations for their training videos from text scripts.",
            "Amazon Polly",
            "Amazon Polly turns text into lifelike speech, allowing you to create applications that talk and build entirely new categories of speech-enabled products.",
            ["Amazon Transcribe", "Amazon Lex", "Amazon Connect"]
        ),
        (
            "A news organization wants to analyze articles to identify key phrases, sentiment, and entities mentioned.",
            "Amazon Comprehend",
            "Amazon Comprehend uses NLP to find insights and relationships in text, including sentiment analysis, entity recognition, and key phrase extraction.",
            ["Amazon Textract", "Amazon Kendra", "Amazon Rekognition"]
        ),
        (
            "A manufacturing company wants to implement predictive maintenance by analyzing data from their equipment sensors.",
            "Amazon Lookout for Equipment",
            "Amazon Lookout for Equipment analyzes sensor data to detect abnormal equipment behavior, helping identify potential failures before they occur.",
            ["Amazon Forecast", "Amazon Rekognition", "Amazon SageMaker"]
        ),
        (
            "A retailer wants to forecast product demand for the upcoming holiday season based on historical sales data.",
            "Amazon Forecast",
            "Amazon Forecast uses machine learning to deliver highly accurate forecasts based on historical time-series data.",
            ["Amazon Personalize", "Amazon Comprehend", "Amazon Lookout for Metrics"]
        ),
        (
            "A social media company wants to automatically moderate user-uploaded images and videos to detect inappropriate content.",
            "Amazon Rekognition",
            "Amazon Rekognition includes content moderation capabilities to detect inappropriate, unwanted, or offensive images and videos.",
            ["Amazon Comprehend", "Amazon Macie", "Amazon Textract"]
        ),
        (
            "A company needs to analyze customer support calls to identify common issues and customer sentiment.",
            "Amazon Transcribe Call Analytics",
            "Amazon Transcribe Call Analytics combines automatic speech recognition with natural language processing to transcribe and analyze customer service calls.",
            ["Amazon Connect", "Amazon Comprehend", "Amazon Lex"]
        ),
        (
            "A developer team wants to automatically review their code for quality issues and identify potential optimizations.",
            "Amazon CodeGuru",
            "Amazon CodeGuru provides intelligent recommendations to improve code quality and identify the most expensive lines of code in applications.",
            ["AWS Lambda", "Amazon Q Developer", "Amazon SageMaker"]
        )
    ]
    
    # Start/Restart button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.game_active:
            difficulty = st.selectbox("Select difficulty level:", 
                                      ["Easy (3 options)", "Medium (5 options)", "Hard (7 options)"])
            
            if st.button("Start Game", use_container_width=True, type="primary"):
                st.session_state.game_active = True
                st.session_state.score = 0
                st.session_state.questions_asked = 0
                st.session_state.difficulty = difficulty
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
            incorrect_options = current_scenario[3]
            
            # Create answer options based on difficulty
            if st.session_state.difficulty == "Easy (3 options)":
                num_options = 3
            elif st.session_state.difficulty == "Medium (5 options)":
                num_options = 5
            else:  # Hard
                num_options = 7
                
            # Get additional incorrect options if needed
            all_services = list(set([s[1] for s in scenarios]))  # All possible services
            additional_options = [s for s in all_services if s != correct_answer and s not in incorrect_options]
            
            # Select options based on difficulty
            needed_incorrect = min(num_options - 1, len(incorrect_options))
            selected_incorrect = incorrect_options[:needed_incorrect]
            
            # If we need more options for higher difficulties
            if num_options - 1 > len(selected_incorrect):
                more_needed = num_options - 1 - len(selected_incorrect)
                if more_needed > 0 and additional_options:
                    selected_incorrect += random.sample(additional_options, min(more_needed, len(additional_options)))
            
            # Create final options list and shuffle
            options = [correct_answer] + selected_incorrect
            random.shuffle(options)
            
            # Display scenario
            st.markdown("### Scenario:")
            st.markdown(f"**{scenario_text}**")
            
            # Display options
            st.markdown("### Which AWS AI service is most appropriate for this scenario?")
            
            # Use radio button for service selection
            user_answer = st.radio("Select the best AWS service:", options, key=f"q{st.session_state.questions_asked}")
            
            # Check answer button
            if st.button("Submit Answer", key=f"submit{st.session_state.questions_asked}"):
                if user_answer == correct_answer:
                    st.success("✅ Correct! That's the right service for this scenario.")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. The right service is {correct_answer}.")
                
                # Show explanation
                st.info(f"**Explanation**: {explanation}")
                
                # Set a flag in session state to show the answer was submitted
                st.session_state.answer_submitted = True

            # Move the Next Question button outside the submit conditional
            if getattr(st.session_state, 'answer_submitted', False):
                if st.button("Next Question", key=f"next{st.session_state.questions_asked}"):
                    st.session_state.questions_asked += 1
                    st.session_state.answer_submitted = False  # Reset the flag
                    st.rerun()                
                
                
                # Next question button
                # if st.button("Next Question", key=f"next{st.session_state.questions_asked}"):
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
                st.success("Perfect score! You're an AWS AI services expert!")
            elif final_score >= 8:
                st.success("Great job! You have a strong understanding of AWS AI services!")
            elif final_score >= 6:
                st.info("Good effort! You understand the basics but might want to review some AWS AI services.")
            else:
                st.warning("You might want to review the AWS AI services and their use cases.")
            
            # Display a summary table
            st.markdown("### AWS AI Services Summary")
            
            service_data = [
                {"Service": "Amazon Rekognition", "Use Case": "Image and video analysis, facial recognition, celebrity identification, content moderation"},
                {"Service": "Amazon Textract", "Use Case": "Extract text, data, and tables from scanned documents"},
                {"Service": "Amazon Comprehend", "Use Case": "Natural language processing to extract insights and relationships in text"},
                {"Service": "Amazon Comprehend Medical", "Use Case": "Extract information from unstructured medical text"},
                {"Service": "Amazon Transcribe", "Use Case": "Convert speech to text"},
                {"Service": "Amazon Polly", "Use Case": "Convert text to lifelike speech"},
                {"Service": "Amazon Translate", "Use Case": "Translate text between languages"},
                {"Service": "Amazon Lex", "Use Case": "Build conversational interfaces like chatbots"},
                {"Service": "Amazon Personalize", "Use Case": "Create real-time personalized recommendations"},
                {"Service": "Amazon Forecast", "Use Case": "Time-series forecasting service"},
                {"Service": "Amazon Kendra", "Use Case": "Intelligent search service with natural language understanding"},
                {"Service": "Amazon Fraud Detector", "Use Case": "Identify potentially fraudulent online activities"},
                {"Service": "Amazon CodeGuru", "Use Case": "Automated code reviews and application performance recommendations"},
                {"Service": "Amazon Lookout for Equipment", "Use Case": "Detect abnormal equipment behavior from sensor data"}
            ]
            
            df = pd.DataFrame(service_data)
            st.table(df)
            
            if st.button("Play Again"):
                st.session_state.game_active = False
                st.rerun()

if __name__ == "__main__":
    main()
