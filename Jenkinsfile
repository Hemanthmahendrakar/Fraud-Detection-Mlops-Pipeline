pipeline {
    agent any

    environment {
        DATASET_PATH = "/opt/datasets/training_data.csv"

        IMAGE_NAME = "fraud-mlops"
        IMAGE_TAG  = "latest"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/Hemanthmahendrakar/Fraud-Detection-Mlops-Pipeline.git'
            }
        }

        stage('Debug Environment') {
            steps {
                sh '''
                    echo "========== DEBUG =========="

                    pwd

                    docker --version

                    echo ""
                    echo "Dataset"

                    echo "$DATASET_PATH"

                    if [ ! -f "$DATASET_PATH" ]; then
                        echo "Dataset Missing"
                        exit 1
                    fi

                    ls -lh "$DATASET_PATH"

                    mkdir -p models
                    mkdir -p logs
                    mkdir -p artifacts
                    mkdir -p mlruns

                    ls -al
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                '''
            }
        }

        stage('Run MLOps Pipeline') {
            steps {
                sh '''
                    docker rm -f fraud-mlops-container || true

                    docker run --name fraud-mlops-container \
                        --network host \
                        -e DATASET_PATH=${DATASET_PATH} \
                        -v /opt/datasets:/opt/datasets \
                        -v $WORKSPACE/models:/app/models \
                        -v $WORKSPACE/logs:/app/logs \
                        -v $WORKSPACE/artifacts:/app/artifacts \
                        -v $WORKSPACE/mlruns:/app/mlruns \
                        ${IMAGE_NAME}:${IMAGE_TAG}
                '''
            }
        }
    }

    post {

        success {
            echo 'Docker Pipeline Executed Successfully'
        }

        failure {
            echo 'Docker Pipeline Failed'
        }

        always {
            sh 'docker rm -f fraud-mlops-container || true'
            cleanWs()
        }
    }
}
