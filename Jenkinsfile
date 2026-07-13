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

                    mkdir -p models
                    mkdir -p logs
                    mkdir -p artifacts
                    mkdir -p mlruns

                    echo ""
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

        stage('Build Docker Image') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub_cred',
                        usernameVariable: 'DOCKER_USERNAME',
                        passwordVariable: 'DOCKER_PASSWORD'
                    )
                ]) {

                    sh '''
                        IMAGE_NAME=$DOCKER_USERNAME/fraud-detection-mlops
                        IMAGE_TAG=v${BUILD_NUMBER}

                        docker build -t $IMAGE_NAME:$IMAGE_TAG .
                    '''
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub_cred',
                        usernameVariable: 'DOCKER_USERNAME',
                        passwordVariable: 'DOCKER_PASSWORD'
                    )
                ]) {

                    sh '''
                        IMAGE_NAME=$DOCKER_USERNAME/fraud-detection-mlops
                        IMAGE_TAG=v${BUILD_NUMBER}

                        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

                        docker push $IMAGE_NAME:$IMAGE_TAG

                        docker logout
                    '''
                }
            }
        }

        stage('Update Kubernetes Manifest') {
            steps {
                script {

                    sh """
                    sed -i 's|image: hemanthmahendrakar/fraud-detection-mlops:.*|image: hemanthmahendrakar/fraud-detection-mlops:v${BUILD_NUMBER}|' k8s/deployment.yaml
                    """

                    withCredentials([usernamePassword(
                        credentialsId: 'github-creds',
                        usernameVariable: 'GIT_USER',
                        passwordVariable: 'GIT_PASS'
                    )]) {

                        sh """
                        git config user.name "Jenkins"

                        git config user.email "jenkins@local"

                        git add k8s/deployment.yaml

                        git commit -m "Updated image to v${BUILD_NUMBER}" || true

                        git push https://${GIT_USER}:${GIT_PASS}@github.com/Hemanthmahendrakar/Fraud-Detection-Mlops-Pipeline.git HEAD:main
                        """
                    }
                }
            }
        }

    }

    post {

        success {
            echo "======================================"
            echo "Pipeline Executed Successfully"
            echo "Docker Image Version : v${BUILD_NUMBER}"
            echo "======================================"
        }

        failure {
            echo "Pipeline Failed"
        }

        always {
            cleanWs()
        }
    }
}
