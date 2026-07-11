pipeline {
    agent any

    environment {
        DATASET_PATH = "/opt/datasets/training_data.csv"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/Hemanthmahendrakar/Fraud-Detection-Mlops-Pipeline.git'
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh '''
                    python3 -m venv venv

                    . venv/bin/activate

                    python -m pip install --upgrade pip setuptools wheel

                    pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('Debug Environment') {
            steps {
                sh '''
                    echo "========== DEBUG =========="

                    pwd

                    python3 --version

                    echo ""
                    echo "Dataset"

                    echo "$DATASET_PATH"

                    if [ ! -f "$DATASET_PATH" ]; then
                        echo "Dataset Missing"
                        exit 1
                    fi

                    ls -lh "$DATASET_PATH"

                    echo ""

                    mkdir -p models
                    mkdir -p logs
                    mkdir -p artifacts
                    mkdir -p mlruns

                    echo "Workspace"

                    ls -al
                '''
            }
        }

        stage('Run MLOps Pipeline') {
            steps {
                sh '''
                    . venv/bin/activate

                    mkdir -p models
                    mkdir -p logs
                    mkdir -p artifacts
                    mkdir -p mlruns

                    python pipeline.py
                '''
            }
        }
    }

    post {

        success {
            echo 'Pipeline Executed Successfully'
        }

        failure {
            echo 'Pipeline Failed'
        }

        always {
            cleanWs()
        }
    }
}
