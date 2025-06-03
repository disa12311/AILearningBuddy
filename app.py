import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_manager import DataManager
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer
from utils.data_optimizer import DataOptimizer
from database.db_service import DatabaseService

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'data_optimizer' not in st.session_state:
    st.session_state.data_optimizer = DataOptimizer()
if 'db_service' not in st.session_state:
    try:
        st.session_state.db_service = DatabaseService()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.session_state.db_service = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'model_comparison_data' not in st.session_state:
    st.session_state.model_comparison_data = []

def main():
    st.set_page_config(
        page_title="Self-Learning AI Application",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Self-Learning AI Application")
    st.markdown("**Train machine learning models offline without external APIs**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "📊 Data Management",
        "🔧 Data Optimization",
        "🤖 Model Training",
        "🔮 Make Predictions",
        "📈 Model Performance",
        "📚 Learning History",
        "💾 Database Analytics"
    ])
    
    if page == "📊 Data Management":
        data_management_page()
    elif page == "🔧 Data Optimization":
        data_optimization_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "🔮 Make Predictions":
        prediction_page()
    elif page == "📈 Model Performance":
        performance_page()
    elif page == "📚 Learning History":
        history_page()
    elif page == "💾 Database Analytics":
        database_analytics_page()

def data_management_page():
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload Data", "View Data", "Generate Sample Data"])
    
    with tab1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data_manager.set_data(df)
                st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")
                st.dataframe(df.head())
                
                # Basic data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with tab2:
        st.subheader("Current Dataset")
        current_data = st.session_state.data_manager.get_data()
        
        if current_data is not None:
            st.dataframe(current_data)
            
            # Data statistics
            if st.checkbox("Show Statistical Summary"):
                st.subheader("Statistical Summary")
                st.dataframe(current_data.describe())
                
            # Data visualization
            if st.checkbox("Show Data Visualization"):
                st.subheader("Data Visualization")
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols)
                    
                    if x_col and y_col:
                        fig = px.scatter(current_data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📁 No data loaded yet. Please upload a dataset or generate sample data.")

def data_optimization_page():
    st.header("🔧 Data Optimization")
    
    current_data = st.session_state.data_manager.get_data()
    
    if current_data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Management page.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "Feature Engineering", "Outlier Detection", "Missing Values"])
    
    with tab1:
        st.subheader("Data Quality Report")
        
        if st.button("Generate Quality Report"):
            with st.spinner("Analyzing data quality..."):
                report = st.session_state.data_optimizer.get_data_quality_report(current_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", report['basic_info']['shape'][0])
                    st.metric("Total Columns", report['basic_info']['shape'][1])
                with col2:
                    st.metric("Memory Usage (MB)", f"{report['basic_info']['memory_usage_mb']:.2f}")
                    st.metric("Duplicate Rows", report['basic_info']['duplicates'])
                with col3:
                    st.metric("Missing Values", report['missing_values']['total_missing'])
                
                if report['missing_values']['total_missing'] > 0:
                    st.subheader("Missing Values by Column")
                    missing_df = pd.DataFrame([
                        {'Column': col, 'Missing Count': count, 'Missing %': f"{pct:.1f}%"}
                        for col, count, pct in zip(
                            report['missing_values']['missing_by_column'].keys(),
                            report['missing_values']['missing_by_column'].values(),
                            report['missing_values']['missing_percentage'].values()
                        ) if count > 0
                    ])
                    st.dataframe(missing_df)
    
    with tab2:
        st.subheader("Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Selection**")
            selection_method = st.selectbox("Selection Method", 
                ["auto", "univariate", "rfe", "model_based"])
            
            if selection_method != "auto":
                k_features = st.slider("Number of Features", 1, min(20, len(current_data.columns)-1), 10)
            else:
                k_features = 10
            
            if st.button("Apply Feature Selection"):
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    # Use last column as target for demo
                    X = current_data[numeric_cols[:-1]]
                    y = current_data[numeric_cols[-1]]
                    
                    X_selected, selected_features = st.session_state.data_optimizer.feature_selection(
                        X, y, method=selection_method, k=k_features, problem_type='regression'
                    )
                    
                    st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
                else:
                    st.warning("Need at least 2 numeric columns for feature selection.")
        
        with col2:
            st.write("**Feature Scaling**")
            scaling_method = st.selectbox("Scaling Method", 
                ["standard", "minmax", "robust", "power"])
            
            if st.button("Apply Scaling"):
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    X_scaled = st.session_state.data_optimizer.scale_features(
                        current_data[numeric_cols], method=scaling_method
                    )
                    st.success(f"Applied {scaling_method} scaling to numeric features.")
                    st.dataframe(X_scaled.head())
                else:
                    st.warning("No numeric columns found for scaling.")
    
    with tab3:
        st.subheader("Outlier Detection")
        
        outlier_method = st.selectbox("Detection Method", ["iqr", "zscore"])
        
        if st.button("Detect Outliers"):
            outliers = st.session_state.data_optimizer.detect_outliers(current_data, method=outlier_method)
            
            total_outliers = sum(len(indices) for indices in outliers.values())
            st.metric("Total Outliers Detected", total_outliers)
            
            if total_outliers > 0:
                outlier_summary = []
                for col, indices in outliers.items():
                    if len(indices) > 0:
                        outlier_summary.append({
                            'Column': col,
                            'Outlier Count': len(indices),
                            'Outlier %': f"{len(indices)/len(current_data)*100:.1f}%"
                        })
                
                if outlier_summary:
                    st.dataframe(pd.DataFrame(outlier_summary))
                
                if st.button("Remove Outliers"):
                    cleaned_data = st.session_state.data_optimizer.remove_outliers(current_data, method=outlier_method)
                    st.session_state.data_manager.set_data(cleaned_data)
                    st.success(f"Removed outliers. Dataset size: {len(current_data)} → {len(cleaned_data)}")
                    st.rerun()
    
    with tab4:
        st.subheader("Missing Value Handling")
        
        missing_strategy = st.selectbox("Imputation Strategy", 
            ["auto", "knn", "iterative"])
        
        if st.button("Handle Missing Values"):
            if current_data.isnull().sum().sum() > 0:
                with st.spinner("Processing missing values..."):
                    cleaned_data = st.session_state.data_optimizer.handle_missing_values(
                        current_data, strategy=missing_strategy
                    )
                    st.session_state.data_manager.set_data(cleaned_data)
                    
                    original_missing = current_data.isnull().sum().sum()
                    new_missing = cleaned_data.isnull().sum().sum()
                    
                    st.success(f"Missing values: {original_missing} → {new_missing}")
                    st.rerun()
            else:
                st.info("No missing values detected in the dataset.")
        
        col1, col2 = st.columns(2)
        with col1:
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
        with col2:
            n_samples = st.number_input("Number of Samples", min_value=50, max_value=2000, value=200)
        
        if st.button("Generate Sample Data"):
            sample_data = st.session_state.data_manager.generate_sample_data(problem_type.lower(), n_samples)
            st.session_state.data_manager.set_data(sample_data)
            st.success(f"✅ Generated {problem_type.lower()} dataset with {n_samples} samples!")
            st.dataframe(sample_data.head())

def model_training_page():
    st.header("🤖 Model Training")
    
    current_data = st.session_state.data_manager.get_data()
    
    if current_data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Management page.")
        return
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        # Select target column
        target_col = st.selectbox("Target Column", current_data.columns.tolist())
        
        # Select feature columns
        available_features = [col for col in current_data.columns if col != target_col]
        feature_cols = st.multiselect("Feature Columns", available_features, default=available_features)
        
        # Determine problem type
        if pd.api.types.is_numeric_dtype(current_data[target_col]):
            unique_values = current_data[target_col].nunique()
            if unique_values <= 10:
                problem_type = st.selectbox("Problem Type", ["Classification", "Regression"], index=0)
            else:
                problem_type = st.selectbox("Problem Type", ["Regression", "Classification"], index=0)
        else:
            problem_type = "Classification"
            st.info(f"Detected categorical target. Problem type set to: {problem_type}")
        
        # Select algorithm
        if problem_type == "Classification":
            algorithm = st.selectbox("Algorithm", [
                "Random Forest",
                "Gradient Boosting",
                "AdaBoost",
                "Neural Network",
                "Support Vector Machine",
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors",
                "Naive Bayes"
            ])
        else:
            algorithm = st.selectbox("Algorithm", [
                "Random Forest",
                "Gradient Boosting",
                "AdaBoost",
                "Neural Network",
                "Support Vector Machine",
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "ElasticNet",
                "Decision Tree",
                "K-Nearest Neighbors"
            ])
    
    with col2:
        st.subheader("Training Parameters")
        
        # Test size
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        
        # Random state
        random_state = st.number_input("Random State", min_value=0, max_value=1000, value=42)
        
        # Auto-tuning option
        use_auto_tuning = st.checkbox("Auto-tune Hyperparameters", help="Automatically find best parameters using grid search")
        
        # Algorithm-specific parameters
        params = {}
        if not use_auto_tuning:
            if algorithm == "Random Forest":
                params['n_estimators'] = st.slider("Number of Trees", 10, 200, 100)
                params['max_depth'] = st.slider("Max Depth", 1, 20, 10)
                params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
            elif algorithm == "Gradient Boosting":
                params['n_estimators'] = st.slider("Number of Estimators", 50, 200, 100)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                params['max_depth'] = st.slider("Max Depth", 1, 10, 3)
            elif algorithm == "AdaBoost":
                params['n_estimators'] = st.slider("Number of Estimators", 10, 100, 50)
                params['learning_rate'] = st.slider("Learning Rate", 0.1, 2.0, 1.0)
            elif algorithm == "Neural Network":
                layer_size = st.slider("Hidden Layer Size", 50, 200, 100)
                params['hidden_layer_sizes'] = (layer_size,)
                params['learning_rate'] = st.slider("Learning Rate", 0.001, 0.1, 0.001)
                params['max_iter'] = st.slider("Max Iterations", 200, 1000, 500)
            elif algorithm == "Support Vector Machine":
                params['C'] = st.slider("C Parameter", 0.1, 10.0, 1.0)
                params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                params['gamma'] = st.selectbox("Gamma", ['scale', 'auto'])
            elif algorithm == "Logistic Regression":
                params['C'] = st.slider("C Parameter", 0.1, 10.0, 1.0)
                params['penalty'] = st.selectbox("Penalty", ['l2', 'l1'])
            elif algorithm in ["Ridge Regression", "Lasso Regression", "ElasticNet"]:
                params['alpha'] = st.slider("Alpha", 0.1, 10.0, 1.0)
                if algorithm == "ElasticNet":
                    params['l1_ratio'] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
            elif algorithm == "K-Nearest Neighbors":
                params['n_neighbors'] = st.slider("Number of Neighbors", 1, 20, 5)
                params['weights'] = st.selectbox("Weights", ['uniform', 'distance'])
            elif algorithm == "Decision Tree":
                params['max_depth'] = st.slider("Max Depth", 1, 20, 10)
                params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
    
    # Training section
    st.subheader("Train Model")
    
    if len(feature_cols) == 0:
        st.error("❌ Please select at least one feature column.")
        return
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = current_data[feature_cols]
                y = current_data[target_col]
                
                # Check for missing values
                if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                    st.error("❌ Data contains missing values. Please clean your data first.")
                    return
                
                # Auto-tune hyperparameters if selected
                if use_auto_tuning:
                    with st.spinner("Auto-tuning hyperparameters..."):
                        X_processed, y_processed = st.session_state.model_trainer.preprocess_data(X, y, problem_type.lower())
                        best_params, best_score = st.session_state.model_trainer.auto_tune_hyperparameters(
                            X_processed, y_processed, algorithm, problem_type.lower()
                        )
                        params = best_params
                        st.success(f"✅ Best parameters found! Score: {best_score:.4f}")
                        st.json(best_params)
                
                # Train model
                model_id, results = st.session_state.model_trainer.train_model(
                    X, y, algorithm, problem_type.lower(), test_size, random_state, params
                )
                
                # Store model
                st.session_state.trained_models[model_id] = {
                    'model': results['model'],
                    'algorithm': algorithm,
                    'problem_type': problem_type,
                    'feature_cols': feature_cols,
                    'target_col': target_col,
                    'metrics': results['metrics'],
                    'timestamp': datetime.now()
                }
                
                # Save to database
                if st.session_state.db_service:
                    try:
                        # Save model to database
                        st.session_state.db_service.save_model(
                            model_id=model_id,
                            name=f"{algorithm} Model",
                            algorithm=algorithm,
                            problem_type=problem_type.lower(),
                            hyperparameters=params,
                            feature_columns=feature_cols,
                            target_column=target_col,
                            is_auto_tuned=use_auto_tuning,
                            model_file_path=f"models/{model_id}.pkl"
                        )
                        
                        # Save training session
                        cv_scores = {
                            'mean': results['metrics'].get('cv_mean', 0),
                            'std': results['metrics'].get('cv_std', 0)
                        }
                        
                        st.session_state.db_service.save_training_session(
                            model_id=model_id,
                            dataset_id=1,  # Default dataset ID
                            algorithm=algorithm,
                            problem_type=problem_type.lower(),
                            test_size=test_size,
                            random_state=random_state,
                            training_samples=len(results['X_train']),
                            test_samples=len(results['X_test']),
                            metrics=results['metrics'],
                            cv_scores=cv_scores,
                            training_duration=0.0,
                            notes=f"Auto-tuned: {use_auto_tuning}"
                        )
                    except Exception as e:
                        st.warning(f"Failed to save to database: {e}")
                
                # Add to history
                st.session_state.training_history.append({
                    'model_id': model_id,
                    'algorithm': algorithm,
                    'problem_type': problem_type,
                    'timestamp': datetime.now(),
                    'metrics': results['metrics']
                })
                
                st.success(f"✅ Model trained successfully! Model ID: {model_id}")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Training Results")
                    for metric, value in results['metrics'].items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                
                with col2:
                    st.subheader("Model Information")
                    st.info(f"**Algorithm:** {algorithm}")
                    st.info(f"**Problem Type:** {problem_type}")
                    st.info(f"**Features:** {len(feature_cols)}")
                    st.info(f"**Training Samples:** {len(results['X_train'])}")
                    st.info(f"**Test Samples:** {len(results['X_test'])}")
                
                # Advanced Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    if problem_type.lower() == "classification":
                        st.subheader("Confusion Matrix")
                        fig = st.session_state.visualizer.plot_confusion_matrix(
                            results['y_test'], results['y_pred']
                        )
                        st.pyplot(fig)
                    else:
                        st.subheader("Prediction vs Actual")
                        fig = st.session_state.visualizer.plot_regression_results(
                            results['y_test'], results['y_pred']
                        )
                        st.pyplot(fig)
                
                with col2:
                    # Feature importance plot
                    if hasattr(results['model'], 'feature_importances_'):
                        st.subheader("Feature Importance")
                        fig = st.session_state.visualizer.plot_feature_importance(
                            results['model'], feature_cols
                        )
                        if fig:
                            st.pyplot(fig)
                    else:
                        # Learning curve for models without feature importance
                        st.subheader("Learning Curve")
                        try:
                            train_sizes, train_scores, val_scores = st.session_state.model_trainer.generate_learning_curve(
                                X, y, algorithm, problem_type.lower(), params
                            )
                            fig = st.session_state.visualizer.plot_learning_curve(
                                train_scores, val_scores, train_sizes
                            )
                            st.pyplot(fig)
                        except:
                            st.info("Learning curve not available for this model.")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def prediction_page():
    st.header("🔮 Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        return
    
    # Select model
    model_ids = list(st.session_state.trained_models.keys())
    selected_model_id = st.selectbox("Select Model", model_ids)
    
    if selected_model_id:
        model_info = st.session_state.trained_models[selected_model_id]
        
        # Display model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Algorithm:** {model_info['algorithm']}")
        with col2:
            st.info(f"**Type:** {model_info['problem_type']}")
        with col3:
            st.info(f"**Features:** {len(model_info['feature_cols'])}")
        
        st.subheader("Input Features")
        
        # Create input fields for each feature
        feature_values = {}
        current_data = st.session_state.data_manager.get_data()
        
        cols = st.columns(2)
        for i, feature in enumerate(model_info['feature_cols']):
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(current_data[feature]):
                    min_val = float(current_data[feature].min())
                    max_val = float(current_data[feature].max())
                    default_val = float(current_data[feature].mean())
                    feature_values[feature] = st.number_input(
                        f"{feature}", 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default_val,
                        key=f"input_{feature}"
                    )
                else:
                    unique_values = current_data[feature].unique().tolist()
                    feature_values[feature] = st.selectbox(
                        f"{feature}", 
                        unique_values,
                        key=f"input_{feature}"
                    )
        
        # Make prediction
        if st.button("🎯 Make Prediction", type="primary"):
            try:
                # Prepare input data
                input_data = pd.DataFrame([feature_values])
                
                # Make prediction
                model = model_info['model']
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if model_info['problem_type'] == "Classification":
                    st.success(f"**Predicted Class:** {prediction}")
                    
                    # Show prediction probabilities if available
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(input_data)[0]
                        classes = model.classes_
                        
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        st.subheader("Class Probabilities")
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   title="Prediction Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success(f"**Predicted Value:** {prediction:.4f}")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

def performance_page():
    st.header("📈 Model Performance")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    
    comparison_data = []
    for model_id, model_info in st.session_state.trained_models.items():
        row = {
            'Model ID': model_id,
            'Algorithm': model_info['algorithm'],
            'Type': model_info['problem_type'],
            'Timestamp': model_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        }
        row.update(model_info['metrics'])
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Performance visualization
        st.subheader("Performance Visualization")
        
        # Select metric for visualization
        metric_cols = [col for col in comparison_df.columns 
                      if col not in ['Model ID', 'Algorithm', 'Type', 'Timestamp']]
        
        if metric_cols:
            selected_metric = st.selectbox("Select Metric", metric_cols)
            
            fig = px.bar(comparison_df, x='Model ID', y=selected_metric, 
                        color='Algorithm', title=f"{selected_metric} by Model")
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed model analysis
    st.subheader("Detailed Model Analysis")
    
    model_ids = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model for Detailed Analysis", model_ids)
    
    if selected_model:
        model_info = st.session_state.trained_models[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Details:**")
            st.json({
                'Algorithm': model_info['algorithm'],
                'Problem Type': model_info['problem_type'],
                'Features': model_info['feature_cols'],
                'Target': model_info['target_col']
            })
        
        with col2:
            st.write("**Performance Metrics:**")
            for metric, value in model_info['metrics'].items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")

def history_page():
    st.header("📚 Learning History")
    
    if not st.session_state.training_history:
        st.info("📝 No training history available yet. Train some models to see the learning progress!")
        return
    
    # Training timeline
    st.subheader("Training Timeline")
    
    history_df = pd.DataFrame(st.session_state.training_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Timeline visualization
    fig = px.timeline(
        history_df,
        x_start='timestamp',
        x_end='timestamp',
        y='algorithm',
        color='problem_type',
        title="Model Training Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Learning progress
    st.subheader("Learning Progress")
    
    # Extract metrics for plotting
    metrics_data = []
    for i, entry in enumerate(st.session_state.training_history):
        row = {'Training Session': i+1, 'Algorithm': entry['algorithm']}
        row.update(entry['metrics'])
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot learning curves
    metric_cols = [col for col in metrics_df.columns 
                  if col not in ['Training Session', 'Algorithm']]
    
    if metric_cols:
        selected_metric = st.selectbox("Select Metric to Track", metric_cols)
        
        fig = px.line(metrics_df, x='Training Session', y=selected_metric, 
                     color='Algorithm', title=f"{selected_metric} Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Training summary
    st.subheader("Training Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models Trained", len(st.session_state.training_history))
    
    with col2:
        algorithms_used = history_df['algorithm'].nunique()
        st.metric("Algorithms Used", algorithms_used)
    
    with col3:
        problem_types = history_df['problem_type'].nunique()
        st.metric("Problem Types", problem_types)
    
    with col4:
        if len(st.session_state.training_history) > 0:
            latest_training = max(entry['timestamp'] for entry in st.session_state.training_history)
            st.metric("Last Training", latest_training.strftime('%Y-%m-%d'))
    
    # Detailed history table
    st.subheader("Detailed History")
    display_history = history_df.copy()
    display_history['timestamp'] = display_history['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_history)

if __name__ == "__main__":
    main()
