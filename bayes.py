import arff
import sys
import numpy as np
import pandas as pd
import math
from pandas import Series, DataFrame
from collections import OrderedDict


# Attributes list 
def _attribute_list(trainSet):
    # Convert the attributes dictionary to list
    dictionaryList = list(trainSet['attributes'])
   
    for index in range(0,len(dictionaryList) - 1):
        print(dictionaryList[index][0], " class")


# Class dictionary 
def _class_Dic(trainSet):
    
    # Class dictionary declaration 
    cDic = {}
    
    # Initialize the two classes indexes
    index1 = 0
    index2 = 0
 
    # Class dictionary key values define
    key1 = trainSet['attributes'][-1][1][0]
    key2 = trainSet['attributes'][-1][1][1]
   
    # Class number enumeration 
    for i in range(0, len(trainSet['data'])):
        if key1 == trainSet['data'][i][-1]:
               index1 += 1
        else:
           index2 += 1

    # Class dictionary Initialization
    cDic[key1] = index1
    cDic[key2] = index2
    
    return cDic 


# Number of first and second keys in data set
def _num_first_second_key(trainSet):
    
    # Convert the attributes part into list
    attList = list(trainSet['attributes'])
    
    # Key1 and Key2
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]

    # Convert the data part into array
    dataArr = np.array(trainSet['data'])
    
    # counter for key1 and key2
    counter1 = 0
    counter2 = 0

    # Traverse all the data 
    for i in range(0,len(dataArr)):
        if(dataArr[i][-1] == key1):
           counter1 += 1
        if(dataArr[i][-1] == key2):
           counter2 += 1

    return counter1, counter2



# The first class key probability   
def _p_class_agv1(dic):
    
    # Convert dictionary to list
    dicList = list(dic)

    # First value of key
    val1 = dic[dicList[0]]

    # Second value of key
    val2 = dic[dicList[1]]

    return (val1+1)/(val1+val2+2)

# The first class key 
def _class_agv1(dic):
    
    # Convert dictionary to list
    dicList = list(dic)

    return dicList[0]

# The second class key
def _class_agv2(dic):

    # Convert dictionary to list
    dicList = list(dic)

    return dicList[1]

# The first column given first class key value dictionary
def _first_column_attribute_given_first_class_key_dictionary(trainSet):
    
    # First class key
    firstKey = _class_agv1(_class_Dic(trainSet))
   
    # First column attribute dictionary declaration
    attributeDic = {}

    # First column attribute given first class list declaration
    firstAttList = list()

    for item in trainSet['data']: 
        if item[-1] == firstKey:
           firstAttList.append(item[0])

       
    # Frist column attribute given first class array definition
    firstAttArray = np.array(firstAttList)
    
    # First column attribute given first class array uniqueness
    obj = Series(firstAttArray)
    uniques = obj.unique()

    # First column attribute given first class summary
    summary = obj.value_counts()

    # First column attribute given first class dictionary initialization
    for i in range(0,len(uniques)):
        attributeDic[summary.keys()[i]] = summary[i]    
 
    return attributeDic
# The first column given second class key value dictionary
def _first_column_attribute_given_second_class_key_dictionary(trainSet):

    # First class key
    secondKey = _class_agv2(_class_Dic(trainSet))
    
    # First column attribute dictionary declaration
    attributeDic = {}    

    # First column attribute given second class list declaration
    firstAttList = list()
   
    for item in trainSet['data']:
        if item[-1] == secondKey:
           firstAttList.append(item[0])
 
    # First column attribute given first class array definition
    firstAttArray = np.array(firstAttList)

    # Second column attribute given second class array uniqueness
    obj = Series(firstAttArray)
    uniques = obj.unique()
   
    # First column attribute given second class summary
    summary = obj.value_counts()

    # First column attribute given second class dictionary initialization
    for i in range(0,len(uniques)):
        attributeDic[summary.keys()[i]] = summary[i]

    return attributeDic

# Remove the first column of the data set
def _remove_first_column(trainSet): 
 
   for i in trainSet['data']:
       del i[0]
       

# All the attributes dictionaries stored in list given first class key
def _all_attributes_given_first_classKey_list(trainSet):
   
    # Define the attributes dictionaries list
    allAttList = list()
    
    # Column index
    index = 0
    
    # Initialize the count index restriction
    restriction = len(trainSet['data'][0]) - 1 
     
    while(index < restriction):
                    
         firstColumnAtt =  _first_column_attribute_given_first_class_key_dictionary(trainSet)
         allAttList.append(firstColumnAtt)
         _remove_first_column(trainSet)
         index+=1
    
    #Check whether all attributes instance in the attributes given first class dictionary
    for attributeIndex in range(0,len(trainSet['attributes'])):
        for allAttListIndex in range(0,len(allAttList)):
            if(attributeIndex == allAttListIndex):
               for instanceIndex in range(0,len(trainSet['attributes'][attributeIndex][1])):
                   attributeDicList = list(allAttList[allAttListIndex])
                   if(trainSet['attributes'][attributeIndex][1][instanceIndex] not in attributeDicList):
                       allAttList[allAttListIndex][trainSet['attributes'][attributeIndex][1][instanceIndex]] = 0
    return allAttList

# All the attributes dictionaries stored in list given second class key
def _all_attributes_given_second_classKey_list(trainSet):
    
    # Define the attributes dictionaries list
    allAttList = list()

    # Column index
    index = 0
   
    # Initialize the count index restriction
    restriction = len(trainSet['data'][0]) - 1

    while(index < restriction):
         firstColumnAtt = _first_column_attribute_given_second_class_key_dictionary(trainSet)
         allAttList.append(firstColumnAtt) 
         _remove_first_column(trainSet)
         index+=1
    #Check whether all attributes instance in the attributes given first class dictionary
    for attributeIndex in range(0,len(trainSet['attributes'])):
        for allAttListIndex in range(0,len(allAttList)):
            if(attributeIndex == allAttListIndex):
               for instanceIndex in range(0,len(trainSet['attributes'][attributeIndex][1])):
                   attributeDicList = list(allAttList[allAttListIndex])
                   if(trainSet['attributes'][attributeIndex][1][instanceIndex] not in attributeDicList):
                       allAttList[allAttListIndex][trainSet['attributes'][attributeIndex][1][instanceIndex]] = 0
    return allAttList

# All the attributes given first class key dictionaries stored in list with corresponding probability (Laplace Estimation)
def _all_attributes_given_first_classKey_corresponding_Laplace_probability(trainSet):
   
    # All attributes given first class key dictionary list 
    allAttList = _all_attributes_given_first_classKey_list(trainSet)
    
    # Iterate all the dictionaries in the attributes list
    for item in allAttList:
        # Convert dictionary to list for indexing
        dicList = list(item)
           
        # Sum all the instance value
        sum = 0
        for index in range(0,len(dicList)):
            sum += item[dicList[index]] 
            # Pseudocount = 1
            #sum += 1
       
        # Append probability part into dictionary attribute 
        for index in range(0,len(dicList)):
            probability = (item[dicList[index]] + 1)/(sum+len(dicList))
            
            # Define attribute dictionary content list
            # Initialize the list
            content = list()
            content.append(item[dicList[index]])
            content.append(probability)
            item[dicList[index]] =  content


    return allAttList

# All the attributes dictionaries given second class key stored in list with corresponding probability (Laplace Estimation)
def _all_attributes_given_second_classKey_corresponding_Laplace_probability(trainSet):

    # All attributes given second class key dictionary list 
    allAttList = _all_attributes_given_second_classKey_list(trainSet)
    
    # Iterate all the dictionaries in the attributes list
    for item in allAttList:
        # Convert dictionary to list for indexing
        dicList = list(item)
    
        # Sum all the instance value
        sum = 0
        for index in range(0,len(dicList)):
            sum += item[dicList[index]]
            # Pseudocount = 1
            #sum += 1
        
        # Append probility part into dictionary attribute
        for index in range(0,len(dicList)):
            probability = (item[dicList[index]] + 1)/(sum+len(dicList))

            # Define attribute dictionary content list
            # Initialize the list
            content = list()
            content.append(item[dicList[index]])
            content.append(probability)
            item[dicList[index]] = content

    return allAttList

# The attributes dictionaries given first class key add the parameter of probability of the data
def _all_attributes_given_first_second_classKey_add_data_probability(trainSet1,trainSet2,trainSet):
    
    # Get all the attributes given first class key with paplace probability 
    # Get all the attributes given second class key with paplace probability 
    allAttList1 = _all_attributes_given_first_classKey_corresponding_Laplace_probability(trainSet1)
    allAttList2 = _all_attributes_given_second_classKey_corresponding_Laplace_probability(trainSet2)
   
    # Iterate all the correspoinding items in attributes list given first class key and attributes given second class key
    for index1 in range(0,len(allAttList1)):
        for index2 in range(0,len(allAttList2)):
            if(index1 == index2):
               dicList1 = list(allAttList1[index1])
               dicList2 = list(allAttList2[index2])

               for dicIndex1 in range(0,len(dicList1)):
                   for dicIndex2 in range(0,len(dicList2)):
                       if(dicList1[dicIndex1] == dicList2[dicIndex2]):
                          content1 = allAttList1[index1][dicList1[dicIndex1]]
                          content2 = allAttList2[index2][dicList2[dicIndex2]]
                          sum = content1[0] + content2[0]
                          allAttList1[index1][dicList1[dicIndex1]].append((sum+1)/(len(trainSet['data'])+len(dicList1)))
                          allAttList2[index2][dicList2[dicIndex2]].append((sum+1)/(len(trainSet['data'])+len(dicList2)))
   
    return allAttList1, allAttList2

# Calculate the Laplace Estimation
def _laplace_estimation(trainSet1, trainSet2, trainSet, testSet):
        
    # The attributes given by frist and second class dictionaries list
    allAttList1, allAttList2 = _all_attributes_given_first_second_classKey_add_data_probability(trainSet1,trainSet2,trainSet)
    
    # Class dictionary
    classDic = _class_Dic(trainSet)
   
    key1 = _class_agv1(classDic)
    key2 = _class_agv2(classDic)
    
    # First class probability 
    # Second class probability
    pClass1 = _p_class_agv1(classDic)
    pClass2 = 1 - pClass1

    # Declare and initialize the prediction dictionary
    predictDic = {}

    # Prediction is correct index
    correct = 0

    # Iterate every line of the test data part
    for item in testSet['data']:
       
           p3 = 1
           p4 = 1
           # Product base for likelihood and probability initialization
           # Data probability
           pBase3 = 1
           pBase4 = 1
           dataBase3 = 1
           dataBase4 = 1
           # Iterate each element in each line
           for indexTest in range(0,len(item)-1):
                 
               # Iterate each instance of the attributes given second class list
               for indexAtt in range(0,len(allAttList1)):
                   if(indexTest == indexAtt):
                      # Iterate the instance of the dictionary
                      # Set the dictionary to list
                      dicList = list(allAttList2[indexAtt])
                      for indexDic in range(0,len(dicList)):
                          # If the instance in the attribute equals the instance of the element in the line
                          if(dicList[indexDic] ==  item[indexTest]):
                             pBase3 *= allAttList1[indexAtt][dicList[indexDic]][1]
                             dataBase3 *= allAttList1[indexAtt][dicList[indexDic]][2]
           p3 = pBase3 * pClass1 / dataBase3
           
           # Iterate each element in each line
           for indexTest in range(0,len(item)-1):
               # Iterate each instance of the attributes given second class list
               for indexAtt in range(0,len(allAttList2)):
                   if(indexTest == indexAtt):
                      # Iterate the instance of the dictionary
                      # Set the dictionary to list
                      dicList = list(allAttList2[indexAtt])
                      for indexDic in range(0,len(dicList)):
                          # If the instance in the attribute equals the instance of the element in the line
                          if(dicList[indexDic] ==  item[indexTest]):
                            pBase4 *= allAttList2[indexAtt][dicList[indexDic]][1]
                            dataBase4 *= allAttList2[indexAtt][dicList[indexDic]][2]
           p4 = pBase4 * pClass2 / dataBase4

           # Posterior probability of first class
           percentage = p3/(p3+p4)
           # Decision
           if(percentage > 0.5):
              if(item[-1] == key1):
                 correct += 1
                 print(key1, " ", key1, " ", percentage)
              else: 
                 print(key1, " ", key2, " ", percentage)
           else:
              if(item[-1] == key1):
                 print(key2, " ", key1, " ", 1-percentage)
              else:
                 correct += 1
                 print(key2, " ", key2, " ", 1 - percentage)
    print()
    print(correct)   


# Define method to get all the attributes pair list
def _attributes_list(trainSet):
    
    # Define the empty attributes list
    attList = list()

    # Convert the attributes dictionary into list
    dicList = list(trainSet['attributes']) 

    # Traverse all the attributes and store the pairs into the list
    for index1 in range(0,len(dicList)-2):
        #while(index != len(dicList)-2):
         #   attList.append(dicList[index][0])
          #  attList.append(dicList[index+1][0])
           # index += 1
         for index2 in range(index1+1,len(dicList)-1):
             attList.append(dicList[index1][0])
             attList.append(dicList[index2][0])    

    return attList


# Define method to get the index of the attributes of the given attributes
def _index_get(attName1, attName2, trainSet):
    
    # Define the attributes list
    attList = list(trainSet['attributes'])

    for i in range(0, len(attList)):
        if(attName1 == attList[i][0]):
           attIndex1 = i
        elif(attName2 == attList[i][0]):
           attIndex2 = i

    return attIndex1, attIndex2

# Define method to get the index of the attribute 
def _att_index_get(attName, trainSet):
    
    # Convert attributes dictionary into list
    attList = list(data['attributes'])

    # Traverse the list
    for i in range(0,len(attList)):
        if(attName == attList[i][0]):
           attIndex = i

    return attIndex



# Define method to get the conditional probability value of all the instance of the attribute
def _get_instance_conditonal_probability_given_first_second_key(attName, trainSet):

    # Define the empty instance CPT list, P(X|Y)
    instance_CPT_key1_List = list()
    instance_CPT_key2_List = list()
 
    # Get the index of the attribute in the list
    attIndex = _att_index_get(attName,trainSet)
    # Convert the data lists into array
    dataArr = np.array(trainSet['data'])       
    
    # Number of the first and second class key number
    numKey1, numKey2 =  _num_first_second_key(trainSet)
    
    # Convert the attributes dictionary into list
    attList = list(trainSet['attributes'])
   
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]

    # Traverse all the instance of the attribute
    for item in attList[attIndex][1]:
        # Counter index for counting the number of the instance given key1 and key2 
        counter1 = 0
        counter2 = 0
        # Traverse all the instance values in the data array
        for i in range(0,len(dataArr)):
            
            if(dataArr[i][attIndex] == item):
               if(dataArr[i][-1] == key1):
                  counter1 += 1
               if(dataArr[i][-1] == key2): 
                  counter2 += 1
           # if(dataArr[i][attIndex] == item and dataArr[i][-1] == key2):
            #   counter2 += 1
        cp1 = (counter1+1)/(numKey1+len(attList[attIndex][1]))
        cp2 = (counter2+1)/(numKey2+len(attList[attIndex][1]))
        instance_CPT_key1_List.append(cp1)
        instance_CPT_key2_List.append(cp2)
                    
               
    return instance_CPT_key1_List, instance_CPT_key2_List # The value are ordered with the sequence of the instance of the attribute


# Define method to get the condition probability of an attribute given another attribute and class, P(Xi|Xj,Y)
def _get_instance_conditional_probability_given_another_attribute_and_class(attName, givenAtt, trainSet):

    # Define the empty instance CPT lists, P(Xi|Xj,Y)
    instance_CPT_instance_key1_list = list()
    instance_CPT_instance_key2_list = list()

    # Get the index of the attribute in the list
    attIndex1 = _att_index_get(attName,trainSet)
    attIndex2 = _att_index_get(givenAtt,trainSet)

    # Convert the data into array
    dataArr = np.array(trainSet['data'])

    # Convert the attributes dictionary into list
    attList = list(trainSet['attributes'])
    
    # Class keys name specification
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]


    # Traverse all the instance of the attribute
    for item1 in attList[attIndex1][1]:
        for item2 in attList[attIndex2][1]:
            # Counter for counting the number of given conditions
            givenCounter1 = 0
            givenCounter2 = 0
            counter1 = 0
            counter2 = 0
            for i in range(0,len(dataArr)):

                if(dataArr[i][attIndex2] == item2 and dataArr[i][-1] == key1):
                   givenCounter1 += 1
                   if(dataArr[i][attIndex1] == item1):
                       counter1 += 1
                if(dataArr[i][attIndex2] == item2 and dataArr[i][-1] == key2):
                   givenCounter2 += 1
                   if(dataArr[i][attIndex1] == item1):
                       counter2 += 1  
            
            cp1 = (counter1 + 1)/(givenCounter1 + len(attList[attIndex1][1]))
            cp2 = (counter2 + 1)/(givenCounter2 + len(attList[attIndex1][1]))
            instance_CPT_instance_key1_list.append(cp1)
            instance_CPT_instance_key2_list.append(cp2)


    return instance_CPT_instance_key1_list, instance_CPT_instance_key2_list
        


# Define the method to get the conditional probability of one instance and another instance given class key,P(Xi,Xj|Y)
def _probability_of_instance_instance_given_first_second_class_key_list(attName1, attName2, trainSet):

    # Define the empty instance CPT lists, P(Xi,Xj|Y)
    instance_instance_given_key1_list = list()
    instance_instance_given_key2_list = list()

    # Get the index of the attribute in the list
    attIndex1 = _att_index_get(attName1,trainSet)
    attIndex2 = _att_index_get(attName2,trainSet)

    # Convert the data into array
    dataArr = np.array(trainSet['data'])

    # Number of the first and second class key number
    numKey1, numKey2 =  _num_first_second_key(trainSet)
    # Convert the attributes dictionary into list
    attList = list(trainSet['attributes'])

    # Class keys name specification
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]

    # Traverse all the instance of the attribute
    for item1 in attList[attIndex1][1]:
        for item2 in attList[attIndex2][1]:
            # Counter for counting the number of given conditions
            counter1 = 0
            counter2 = 0          
  
            for i in range(0,len(dataArr)):

                if(dataArr[i][-1] == key1 and dataArr[i][attIndex1] == item1 and dataArr[i][attIndex2] == item2):
                      counter1 += 1
                if(dataArr[i][-1] == key2 and dataArr[i][attIndex1] == item1 and dataArr[i][attIndex2] == item2):
                      counter2 += 1
            cp1 = (counter1 + 1)/(numKey1 + len(attList[attIndex1][1])*len(attList[attIndex2][1]))                 
            cp2 = (counter2 + 1)/(numKey2 + len(attList[attIndex1][1])*len(attList[attIndex2][1]))                 
            instance_instance_given_key1_list.append(cp1)
            instance_instance_given_key2_list.append(cp2)
    return instance_instance_given_key1_list, instance_instance_given_key2_list           


# Define method to calculate the probability of first instance, second instance, key, P(Xi,Xj,Y)
def _probability_of_instance1_instance2_key(attName1,attName2,trainSet):
    
    # Define the empty instance CPT lists, P(Xi,Xj|Y)
    instance_instance_key1_list = list()
    instance_instance_key2_list = list()

    # Get the index of the attribute in the list
    attIndex1 = _att_index_get(attName1,trainSet)
    attIndex2 = _att_index_get(attName2,trainSet)

    # Convert the data into array
    dataArr = np.array(trainSet['data'])

    # Convert the attributes dictionary into list
    attList = list(trainSet['attributes'])

    # Class keys name specificationis
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]

    # Traverse all the instance of the attribute
    for item1 in attList[attIndex1][1]:
        for item2 in attList[attIndex2][1]:

            # Counter for counting the number of given conditions
            counter1 = 0
            counter2 = 0          
            for i in range(0,len(dataArr)):

                if(dataArr[i][-1] == key1 and dataArr[i][attIndex1] == item1 and dataArr[i][attIndex2] == item2):
                      counter1 += 1
                if(dataArr[i][-1] == key2 and dataArr[i][attIndex1] == item1 and dataArr[i][attIndex2] == item2):
                      counter2 += 1
            cp1 = (counter1 + 1)/(len(dataArr) + len(attList[attIndex1][1])*len(attList[attIndex2][1]*len(attList[-1][1])))             
            cp2 = (counter2 + 1)/(len(dataArr) + len(attList[attIndex1][1])*len(attList[attIndex2][1]*len(attList[-1][1])))             
            instance_instance_key1_list.append(cp1)
            instance_instance_key2_list.append(cp2)
    return instance_instance_key1_list,instance_instance_key2_list                  



# Define conditional mutual information between two attributes
def _conditional_mutual_infomation_between_two_attributes(attName1, attName2, trainSet):



    # P(Xi|Y1),P(Xj|Y2),P(Xj|Y1),P(Xj|Y2)
    instance1_CPT_key1_List, instance1_CPT_key2_List = _get_instance_conditonal_probability_given_first_second_key(attName1, trainSet)
    instance2_CPT_key1_List, instance2_CPT_key2_List = _get_instance_conditonal_probability_given_first_second_key(attName2, trainSet)


    # P(Xi,Xj|Y1), P(Xi,Xj|Y2)
    instance_instance_given_key1_list, instance_instance_given_key2_list = _probability_of_instance_instance_given_first_second_class_key_list(attName1, attName2, trainSet)
    
    # P(Xi,Xj,Y1), P(Xi,Xj,Y2)
    instance_instance_key1_list,instance_instance_key2_list = _probability_of_instance1_instance2_key(attName1,attName2,trainSet)

   
    # Conditional mutual information counter 
    mutual_info_key1_counter = 0
    mutual_info_key2_counter = 0

    for index in range(0,len(instance_instance_key1_list)):
        mutual_info_key1_counter += instance_instance_key1_list[index]*math.log2(instance_instance_given_key1_list[index]/(instance1_CPT_key1_List[int(index/len(instance2_CPT_key1_List))]*instance2_CPT_key1_List[index%len(instance2_CPT_key1_List)])) 
    
    for index in range(0,len(instance_instance_key2_list)):
        mutual_info_key2_counter += instance_instance_key2_list[index]*math.log2(instance_instance_given_key2_list[index]/(instance1_CPT_key2_List[int(index/len(instance2_CPT_key2_List))]*instance2_CPT_key2_List[index%len(instance2_CPT_key2_List)])) 
    
    mutualInfo = mutual_info_key1_counter + mutual_info_key2_counter

    return mutualInfo


# Define method to find out the maximum mutual info for certain attribute from a attributes list
def _find_out_maximum_mutual_info_node(attName, attList,trainSet):

   maxValue = 0  

   for index in range(0,len(attList)):
          
          if(maxValue < _conditional_mutual_infomation_between_two_attributes(attName, attList[index], trainSet)):
             maxValue = _conditional_mutual_infomation_between_two_attributes(attName, attList[index], trainSet)
             maxAtt = attList[index]

   return maxAtt, maxValue



# Define the prime's algorithm to find the maximum spanning tree
def _prime_algorithm_for_maximum_spanning_tree_node(trainSet):

    # Convert the attribute dictionary into list
    attList = list(trainSet['attributes'])

    # Define the empty list to store the nodes 
    # Define the initial nodes list, and specification
    s = list()
    
    nodeList = list()
    for i in range(0,len(attList)-1):
        nodeList.append(attList[i][0])

    # Define the possible edge list to store the maximum mutual infor mation
    max_mutual_info_list = list()

    # Define the edge list
    A = list()

    # Add the first attribute into s
    # Remove the first attribute from node list
    s.append(nodeList[0])
    nodeList.remove(nodeList[0])   

    
    # While the node list is not empty, traverse all the attributes from both list
    while(len(nodeList) != 0):
          maxValue = 0
        
          for itemS in s:
              mAtt, value =  _find_out_maximum_mutual_info_node(itemS, nodeList,trainSet)
              if(maxValue < value):
                 maxValue = value   
                 maxNode = mAtt
                 startNode = itemS
          s.append(maxNode)
          nodeList.remove(maxNode)
          A.append([startNode, maxNode])

    return A


# Define method to outpur the nodes and its parent
def _node_parent(trainSet):
   
    # The maximum spanning tree nodes pairs
    A = _prime_algorithm_for_maximum_spanning_tree_node(trainSet)

    # Convert the attribute part into list
    attList = list(trainSet['attributes'])

    print(attList[0][0], " class")

    for i in range(1,len(attList)-1):
      
        for item in A:
            if(item[1] == attList[i][0]):
               print(attList[i][0], " ", item[0], " class")

    


# Define method to traverse each line of the data
def _TAN_classify(testSet, trainSet):

    # Convert the attribute part into list
    attList = list(trainSet['attributes'])

    # Get the maximum spanning tree edge with direction list
    mstList = _prime_algorithm_for_maximum_spanning_tree_node(trainSet)
    
    # The class keys name specification
    key1 = attList[-1][1][0]
    key2 = attList[-1][1][1]

    # Convert the data part into array
    dataArr = np.array(trainSet['data'])
    # Traverse all the data and calculate the probability of first and second key
    counter = 0
    for i in range(0,len(dataArr)):
        if(dataArr[i][-1] == key1):
           counter += 1

    # Probability of key1 
    pKey1 = (counter + 1)/(len(dataArr) + 2)
    pKey2 = 1 - pKey1

    # Get the probability of first attribute instance given class keys
    root_instance_CPT_key1_List, root_instance_CPT_key2_List = _get_instance_conditonal_probability_given_first_second_key(attList[0][0], trainSet)
    
    # Convert testSet into array
    testArr = np.array(testSet['data'])

    # The counter for correct
    correct = 0


    for i in range(0,len(testArr)):
        # Traverse the list and check the probability of the instance
        for rootIndex in range(0,len(attList[0][1])):
            if(attList[0][1][rootIndex] == testArr[i][0]):
               root_instance_index = rootIndex
    
        # The probability of root instance given first and second key
        pRootKey1 = root_instance_CPT_key1_List[root_instance_index] 
        pRootKey2 = root_instance_CPT_key2_List[root_instance_index] 


        p1 = pKey1*pRootKey1
        p2 = pKey2*pRootKey2
        # Traverse all the items in the mstList
        for item in mstList:
            
            instance_CPT_instance_key1_list, instance_CPT_instance_key2_list = _get_instance_conditional_probability_given_another_attribute_and_class(item[1], item[0], trainSet)
            
            # Get the instance index for the two instance of the attributes
            attIndex = _att_index_get(item[1], trainSet)
            givenAttIndex = _att_index_get(item[0], trainSet)

            for index1 in range(0,len(attList[attIndex][1])):
                if(attList[attIndex][1][index1] == testArr[i][attIndex]):
                   instanceIndex = index1
                  
            for index2 in range(0,len(attList[givenAttIndex][1])):
                if(attList[givenAttIndex][1][index2] == testArr[i][givenAttIndex]):
                   givenAttInstanceIndex = index2
            
            p1*=instance_CPT_instance_key1_list[instanceIndex*len(attList[givenAttIndex][1])+givenAttInstanceIndex]
            p2*=instance_CPT_instance_key2_list[instanceIndex*len(attList[givenAttIndex][1])+givenAttInstanceIndex]
        

        percentage = p1/(p1+p2) 

        if(testArr[i][-1] ==  key1):
           if(percentage > 0.5):
              correct += 1
              print(key1,  " ", key1, " ", format(percentage, '.12f'))
           else:
              print(key2, " ", key1, " ", format(1-percentage, '.12f'))
        else:
           if(percentage > 0.5):
              print(key1, " ", key2, " ", format(percentage, '.12f'))
           else:
              correct += 1
              print(key2, " ", key2, " ", format(1-percentage, '.12f'))
    print()
    print(correct)                 
 


 
# Read the train set 
data = arff.load(open(sys.argv[1], 'r'))

# The first data set used for getting the attributes given first class key dictionary
data1 = arff.load(open(sys.argv[1], 'r'))
# The data set used for getting the attributes given second class key dictionaries
data2 = arff.load(open(sys.argv[1], 'r'))

# Read the test set 
test = arff.load(open(sys.argv[2], 'r'))

model = sys.argv[3]


if(model == 'n'):
   _attribute_list(data)
   print() 
   _laplace_estimation(data1, data2, data, test)


if(model == 't'):
   _node_parent(data)
   print()
   _TAN_classify(test, data)
   


