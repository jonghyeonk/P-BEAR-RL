
library(stringdist)

home = getwd()
setwd(paste0(home, '/output'))
data_see = list.files(getwd())

data_all = c('Small', 'Medium', 'Large', 'Huge' , 'Wide',
             'credit-card', 'mccloud', 'Road_Traffic', 'b17')

data_all = c("Small")

{
for( data in data_all){
rate = '1.00'
for(type in c('edit')){
  result= as.numeric()
  for(i in 1:length(rate)){
  setwd(paste0(home, '/input/encoded_normal'))
  test =  read.csv( paste(data, ".csv", sep=''),T)

  setwd(paste0(home, '/input/encoded_anomaly'))
  dat = read.csv(paste(data, "_", rate[i], ".csv", sep='' ),T)

  setwd(paste0(home, '/output'))
  data_see2 = data_see[grepl(data, data_see)]
  
  for(data_alpha in data_see2){
  align= read.csv(data_alpha ,T)
  align = align[which(!is.element(align$concept.name, c('Start', 'End'))),]
  colnames(align)[1:2] = c('Case', 'Activity')
  colnames(align)[5] = c('refined_patterns')

  anomaly= unique(dat[,c(1,18)])
  anomaly[which(anomaly$type_res_trace == ''), 'type_res_trace'] = 'normal'

  repaired = align
  test = test[which(is.element(test$Case, repaired$Case)),]
  
  x = aggregate(repaired$Activity , by= list(repaired$Case), FUN= function(x){paste(x, collapse = '-')})
  y = aggregate(test$Activity , by= list(test$Case), FUN= function(x){paste(x, collapse = '-')})
  
  reconst4= anomaly
  names(reconst4)[2] = 'label'
  reconst4[which(reconst4$label == ''), 'label'] = 'normal'
  labelpadding = function(x){
    paste(sort(unlist(lapply( unlist(strsplit(x, split ="," )), FUN= function(x){
      gsub("[^a-zA-Z]", "", x)})))
      ,collapse = '+')
  }
  
  reconst4$label =  unlist(lapply( reconst4$label, FUN= function(x){labelpadding(x)}))

  accuracy_pattern =0
  accuracy =1- sum(as.character(x$x)!=as.character(y$x))/nrow(x)

  loc_skip = which(is.element(y$Group.1,reconst4[which(reconst4$label=='skip'),'Case'] ))
  accuracy_skip =1- sum( (as.character(x$x)!=as.character(y$x))[loc_skip] )/length(loc_skip)
  loc_insert = which(is.element(y$Group.1,reconst4[which(reconst4$label=='insert'),'Case'] ))
  accuracy_insert =1- sum( (as.character(x$x)!=as.character(y$x))[loc_insert] )/length(loc_insert)
  loc_moved = which(is.element(y$Group.1,reconst4[which(reconst4$label=='moved'),'Case'] ))
  accuracy_moved =1- sum( (as.character(x$x)!=as.character(y$x))[loc_moved] )/length(loc_moved)
  loc_rework = which(is.element(y$Group.1,reconst4[which(reconst4$label=='rework'),'Case'] ))
  accuracy_rework =1- sum( (as.character(x$x)!=as.character(y$x))[loc_rework] )/length(loc_rework)
  loc_replace = which(is.element(y$Group.1,reconst4[which(reconst4$label=='replace'),'Case'] ))
  accuracy_replace =1- sum( (as.character(x$x)!=as.character(y$x))[loc_replace] )/length(loc_replace)
  loc_normal = which(is.element(y$Group.1,reconst4[which(reconst4$label=='normal'),'Case'] ))
  accuracy_normal =1- sum( (as.character(x$x)!=as.character(y$x))[loc_normal] )/length(loc_normal)
  
  
  reconst5 = reconst4[which(reconst4$label != 'normal'),]
  x = reconst5
  test2 = test[which(is.element(test$Case, x$Case)),]
  y = aggregate(test2$Activity , by= list(test2$Case), FUN= function(x){paste(x, collapse = '>>')})
  test3 = repaired[which(is.element(repaired$Case, x$Case)),]
  x = aggregate(test3$Activity , by= list(test3$Case), FUN= function(x){paste(x, collapse = '>>')})
  
  accuracy_anomaly =1- sum(as.character(x$x)!=as.character(y$x))/nrow(x)    
  
  ## performance on single pattern
  reconst5 = reconst4[which(reconst4$label != 'normal'),]
  x = reconst5
  x1 = x[which( !unlist(lapply(reconst5$label, FUN= function(x){
    grepl('+',x, fixed = TRUE)
  }))) ,]
  
  # test2 : single pattern
  test2 = test[which(is.element(test$Case, x1$Case)),]
  y = aggregate(test2$Activity , by= list(test2$Case), FUN= function(x){paste(x, collapse = '>>')})
  test3 = repaired[which(is.element(repaired$Case, x1$Case)),]
  x2 = aggregate(test3$Activity , by= list(test3$Case), FUN= function(x){paste(x, collapse = '>>')})
  accuracy_single =1- sum(as.character(x2$x)!=as.character(y$x))/nrow(x1)

  x3 = aggregate(test3$refined_patterns , by= list(test3$Case), FUN= function(x){x[1]})
  accuracy_pattern_single =1- sum(x3$x!=x1$label)/nrow(x3)

  ## performance on multiple patterns
  reconst5 = reconst4[which(reconst4$label != 'normal'),]
  x = reconst5
  x1 = x[which( unlist(lapply(reconst5$label, FUN= function(x){
    grepl('+',x, fixed = TRUE)
  }))) ,]
  
  case_multiple = x1$Case
  test2 = test[which(is.element(test$Case, x1$Case)),]
  y = aggregate(test2$Activity , by= list(test2$Case), FUN= function(x){paste(x, collapse = '>>')})
  test3 = repaired[which(is.element(repaired$Case, x1$Case)),]
  x2 = aggregate(test3$Activity , by= list(test3$Case), FUN= function(x){paste(x, collapse = '>>')})
  accuracy_multiple =1- sum(as.character(x2$x)!=as.character(y$x))/nrow(x2)

  x3 = aggregate(test3$refined_patterns , by= list(test3$Case), FUN= function(x){x[1]})
  accuracy_pattern_multiple =1- sum(x3$x!=x1$label)/nrow(x3)
  
  
  # REC
  anomaly= unique(dat[,c(1,18)])
  n.c = unique(anomaly[which(anomaly$type_res_trace==''), 'Case'])
  a.c = unique(anomaly[which(anomaly$type_res_trace!=''), 'Case'])
  
  x= reconst4
  test2 = test[which(is.element(test$Case, x$Case)),]
  
  x.act = repaired$Activity
  y.act = test2$Activity
  z.act = dat$Activity
  
  act.list = unique( c(x.act,y.act,z.act) )
  letter.list = c(letters, LETTERS, 1:9)
  
  if(length(act.list) > length(letter.list)){
    print("Over size problem: act.length > letters ")
  }
  
  
  x.act = apply( data.frame(x.act), 1, FUN= function(x){ letter.list[which(act.list==x)]} )  
  y.act = apply( data.frame(y.act), 1, FUN= function(x){ letter.list[which(act.list==x)]} )  
  z.act = apply( data.frame(z.act), 1, FUN= function(x){ letter.list[which(act.list==x)]} )  
  
  str.x = aggregate(x.act , by= list(repaired$Case), FUN= function(x){paste(x, collapse = '')})
  str.y = aggregate(y.act , by= list(test2$Case), FUN= function(x){paste(x, collapse = '')})
  str.z = aggregate(z.act , by= list(dat$Case), FUN= function(x){paste(x, collapse = '')})
  
  loc_multiple = which( is.element(str.x$Group.1 , case_multiple ))
  
  save = c( data_alpha, rate[i] , ceiling(accuracy_pattern*10000)/10000,  
            ceiling(accuracy_pattern_single*10000)/10000, ceiling(accuracy_pattern_multiple*10000)/10000,
            ceiling(accuracy_skip*10000)/10000,  ceiling(accuracy_insert*10000)/10000, 
            ceiling(accuracy_rework*10000)/10000, ceiling(accuracy_replace*10000)/10000,
            ceiling(accuracy_moved*10000)/10000, ceiling(accuracy_normal*10000)/10000,
            ceiling(accuracy_anomaly*10000)/10000, ceiling(accuracy*10000)/10000,
            ceiling(accuracy_single*10000)/10000, ceiling(accuracy_multiple*10000)/10000,
            mean(stringdist(str.z$x, str.y$x, method = "lv")),
            mean(stringdist(str.x$x, str.y$x, method = "lv")),
            mean(stringdist(str.z$x[loc_normal], str.y$x[loc_normal], method = "lv")),
            mean(stringdist(str.x$x[loc_normal], str.y$x[loc_normal], method = "lv")),
            mean(stringdist(str.z$x[-loc_normal], str.y$x[-loc_normal], method = "lv")),
            mean(stringdist(str.x$x[-loc_normal], str.y$x[-loc_normal], method = "lv")),
            
            mean(stringdist(str.x$x[loc_skip], str.y$x[loc_skip], method = "lv")),
            mean(stringdist(str.x$x[loc_insert], str.y$x[loc_insert], method = "lv")),
            mean(stringdist(str.x$x[loc_rework], str.y$x[loc_rework], method = "lv")),
            mean(stringdist(str.x$x[loc_replace], str.y$x[loc_replace], method = "lv")),
            mean(stringdist(str.x$x[loc_moved], str.y$x[loc_moved], method = "lv")),
            mean(stringdist(str.x$x[loc_multiple], str.y$x[loc_multiple], method = "lv")),
            
            0)
  
  result= rbind(result, save)

}
}

result= data.frame(result)
names(result) = c("Data", "Rate","PatternACC", "PatternACC(single)", "PatternACC(multiple)",
                  'p.skip','p.insert','p.rework','p.replace','p.moved', 'p.normal', 'p.anomaly',
                  "ReconACC", "ReconACC(single)", "ReconACC(multiple)", 
                  "ReconERR.before", "ReconERR.after",
                  "ReconERR.before.normal", "ReconERR.after.normal",
                  "ReconERR.before.anomaly", "ReconERR.after.anomaly",
                  'e.skip','e.insert','e.rework','e.replace','e.moved',"e.multiple",
                  "time")

setwd("C:/Users/Jonghyeon/Downloads/PBAR4py/result_RL")

# write.csv(result, paste0(data, ".csv"), row.names = F)
}
}
}


result
