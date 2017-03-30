load('imdb.mat');
finalPath = 'data/view/'
for i = 1:12:size(images.name,2)
    a = split(images.name(i),'/');
    
 
    %if a(2) == 'train'
    folder = fullfile(finalPath,'list/',char(a(2)),char(a(1)));
    g = split(a(3),'_');
    file = [char(a(1)),'_',char(g(2)),'.txt'];
    finalFile=fullfile(folder,file);
    try
        mkdir(folder)
    catch
    end
    dlmwrite(finalFile,num2str(images.class(i)-1),'delimiter','');
    dlmwrite(finalFile,'12','delimiter','','-append');
    for j= 1:12
        %fileName = [pwd '/data/modelnet40v1/' images.name(i+j-1)];
        fileName = ['/home/dinesh/deva/MVCNN-TensorFlow/data/modelnet40v1/' images.name(i+j-1)];
        dlmwrite(finalFile,fileName,'delimiter','','-append');
    end
    listPath = fullfile(finalPath,[char(a(2)),'_lists.txt']);
    dlmwrite(listPath,[finalFile,' ', num2str(images.class(i))],'delimiter','','-append')
    %end
end