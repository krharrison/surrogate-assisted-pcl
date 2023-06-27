dimensions = 30;
popSize = 30;
bayesEvaluations = 50;

% define the input directory
inputDir = sprintf("SobolSamples\\%dD\\%d\\", dimensions, popSize);

% define the root output directory
baseOutputDir = sprintf("TrainedModels\\%dD\\%d\\", dimensions, popSize);

% get all csv files in the input directory
listing = dir(inputDir + "*.csv");

%get the number of files
numFiles = height(listing);

sz = [numFiles 3];
varTypes = ["string","double","double"];
varNames = ["Function","RMSE","R-Squared"];
modelPerformance = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

% loop through each function
for i = 1:numFiles
    file = listing(i);
    [folder, baseFileName, extension] = fileparts(file.name);
    functionName = baseFileName;
    % define and create the output directory
    outputDir = sprintf("%s%s\\", baseOutputDir, functionName);
    %disable the warning for a directory that already exists
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
    mkdir(outputDir);
    
    % read the input data
    inputFile = sprintf("%s%s.csv", inputDir, functionName);
    trainingData = readtable(inputFile);
    
    % run the Bayesian optimisation for a neural network regression model
    [trainedModel, rmse] = trainBayesianNNModel(trainingData, bayesEvaluations, true, outputDir);
    % create a compact model that does not include the training data
    compactModel = compact(trainedModel.RegressionNeuralNetwork);
    %[trainedModel, rmse] = trainAutoRegressionModel(trainingData);
    
    % save the trainined model in the output directory
    save(outputDir + functionName + "-fullModel.mat", "trainedModel")
    save(outputDir + functionName + "-compactModel.mat", "compactModel")
    save(outputDir + functionName + "-BayesOptResults.mat", "BayesoptResults")
    % get the number of samples from the training data
    numSamples = height(trainingData);
    % run the trained model's prediction function on the training data
    predicted = compactModel.predict(trainingData);
    %predicted = trainedModel.predictFcn(trainingData);
    % extract the true fitness from the training data
    actual = trainingData.fitness;
    
    % calculate R^2
    correlation = corrcoef(predicted, actual);
    R = correlation(2,1);
    RSquared = R*R;
    
    modelPerformance(i, :) = {functionName, rmse, RSquared};

    %%% PLOTTING
    
    % plot the Bayesian optimisation fitness 
    h = figure('units','normalized','outerposition',[0 0 1 1], 'visible','off');
    x = 1:BayesoptResults.NumObjectiveEvaluations;
    plot(x, BayesoptResults.ObjectiveMinimumTrace, "LineWidth", 3, "Color", "black");
    xlabel("Function Evaluations");
    ylabel("Objective Value");
    xlim tight
    ylim tight
    set(findall(gcf,'-property','FontSize'),'FontSize', 40)
    % crop and save as PDF
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(h, outputDir + functionName + '-BayesOptResults','-dpdf','-r0')
    
    % create a resposne plot showing predicted vs. actual for each record
    h = figure('units','normalized','outerposition',[0 0 1 1], 'visible','off');
    scatter(1:numSamples, [actual, predicted], 36, "filled")
    legend('Actual','Predicted','Location','BestOutside');
    xlabel("Record")
    ylabel("Fitness")
    grid on
    xlim tight
    ylim tight
    set(findall(gcf,'-property','FontSize'),'FontSize',40)
    
    % crop and save as PDF
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(h, outputDir + functionName + '-Response','-dpdf','-r0')
    
    % create a plot showing predicted vs. actual as a scatter plot
    % including a perfect 
    h = figure('units','normalized','outerposition',[0 0 1 1], 'visible','off');
    scatter(actual, predicted, 36,  "filled")
    xlabel("Actual")
    ylabel("Predicted")
    hold on
    xmin = min(min(actual,predicted));
    xmax = max(max(actual,predicted));
    x=xmin:(xmax-xmin)/1000:xmax;
    y=x;
    plot(x,y, "LineWidth", 3, "Color", "black")
    legend("Observations", "Perfect Prediction", "Location", "Best")
    grid on
    hold off
    xlim tight
    ylim tight
    set(findall(gcf,'-property','FontSize'),'FontSize',40)
   
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(h, outputDir + functionName + '-ActualVPredicted','-dpdf','-r0')
end

writetable(modelPerformance, baseOutputDir + "model-performance.csv")
