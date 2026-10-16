pipeline {
    agent any

    environment {
        DEPLOY_PATH = '/var/www/imfaceid'
        HEALTH_URL  = 'http://127.0.0.1:8000/health'
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/IMadatov/iM.FaceId.git',
                    branch: 'main',
                    credentialsId: 'im-faceid'
            }
        }

        stage('Sync to deploy path') {
            steps {
                sh '''
                sudo mkdir -p ${DEPLOY_PATH}
                sudo rsync -a --delete \
                  --exclude '.env' \
                  --exclude '.git' \
                  --exclude '.venv' \
                  --exclude 'models' \
                  --exclude 'graphify-out' \
                  ./ ${DEPLOY_PATH}/
                '''
            }
        }

        stage('Build & Deploy') {
            steps {
                sh '''
                cd ${DEPLOY_PATH}
                test -f .env || { echo ".env missing in ${DEPLOY_PATH}"; exit 1; }
                docker compose up --build -d
                '''
            }
        }

        stage('Health check') {
            steps {
                sh '''
                for i in 1 2 3 4 5 6 7 8 9 10; do
                  if curl -fsS "${HEALTH_URL}"; then
                    echo "OK"
                    exit 0
                  fi
                  sleep 3
                done
                echo "Health check failed: ${HEALTH_URL}"
                exit 1
                '''
            }
        }
    }
}
