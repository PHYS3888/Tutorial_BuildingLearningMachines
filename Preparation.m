
numEach = 20;

model_followers = randi(22000,numEach,1)+5000;
sport_followers = randi(26000,numEach,1)+2000;

model_heartRate = randn(numEach,1)*15+75;
sport_heartRate = randn(numEach,1)*10+50;

dataMat = [model_followers,model_heartRate;...
    sport_followers,sport_heartRate];

isModel = false(numEach*2,1);
isModel(1:numEach) = true;

save('ModelSportData.mat','dataMat','isModel')
