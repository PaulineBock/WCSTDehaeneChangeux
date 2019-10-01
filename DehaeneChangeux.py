"""
Implementation of a modified version of WCST test passing NN algorithm

Dehaene and Changeux version
For modified WCST  (36 ambiguous cards)
Milner version disponible

Abreviations:
LT => Long-term component
ST => Short-term component
act => activity

Pauline Bock - Mnemosyne Team (INRIA)
25-06-2019
"""

import sys
import os
import time
import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
#print matrices entirely
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
from WCST import *

def WCST_test(nb_test, path):
    reasoning_list = []
    milner_list = []

    def new_card():
        v_data = [] #list type
        np_data = response_item_Reasoning(nb_dim, nb_features, m_percep, reasoning_list) #Modified WCST version
        #np_data = response_item(nb_dim, nb_features, milner_list) #Use Milner version
            
        #Transform into a vector
        for arr in np_data:
            for e in arr:
                v_data.append(e)

        return np_data, v_data
        
    def inhibition_activation(inhibition_activity, LTinhib, STinhib, output_activity, LTout_to_inhib, STout_to_inhib, Tinhib):
        """
        Computes and returns inhibition activity, which desactivates go and reflexion activity when they are too high.
        """
        noise = np.random.uniform(-0.5, 0.5, 1)
        Winhib = np.multiply(LTinhib, STinhib)
        Wout_to_inhib = np.multiply(LTout_to_inhib, STout_to_inhib)
        
        inhibition_activity = sigmoid(
            inhibition_activity * Winhib +
            np.dot(Wout_to_inhib, output_activity) - Tinhib + noise
        )

        return inhibition_activity

    def reflexion_activation(reflexion_activity, Ltref, STref, confidence_activity, STconf_to_ref, LTconf_to_ref, inhibition_activity, LTinhib_to_ref, STinhib_to_ref, Tref):
        """
        Computes and returns reflexion activity according to inverse of error activity, rules activity and reflexion activity.
        """
        noise = np.random.uniform(-0.5, 0.5, 1)
        Wconf_to_ref = STconf_to_ref * LTconf_to_ref
        Winhib_to_ref = np.multiply(LTinhib_to_ref, STinhib_to_ref)
        Wref = np.multiply(LTref, STref)
      
        reflexion_activity = sigmoid(
            (Wconf_to_ref * confidence_activity + Wref * reflexion_activity + Winhib_to_ref*inhibition_activity) - Tref + noise
        )

        return reflexion_activity

    def memory_activation(memory_activity, data, LTinput_to_mem, LTmemory, STinput_to_mem, STmemory, Tmem_int):
        """
        Computes and returns memory activity, given data input,  memory activity and their respective weights.
        """
        noise = np.random.uniform(-0.5, 0.5, 4)
        Wmemory = np.multiply(LTmemory, STmemory)
        Winput_to_mem = np.multiply(STinput_to_mem, LTinput_to_mem)

        memory_activity = memory_activity.copy()
        #For each cluster assemblies
        memory_activity[0:4] = sigmoid( (np.dot(Wmemory, memory_activity[0:4])+ np.dot(Winput_to_mem, v_data[0:4])) - Tmem_int + noise)
        memory_activity[4:8] = sigmoid( (np.dot(Wmemory, memory_activity[4:8])+ np.dot(Winput_to_mem, v_data[4:8])) - Tmem_int + noise)
        memory_activity[8:12] = sigmoid( (np.dot(Wmemory, memory_activity[8:12])+ np.dot(Winput_to_mem, v_data[8:12])) - Tmem_int + noise)

        return memory_activity
    
    def intention_activation(intention_activity, memory_activity, LTintention, STintention, LTmem_to_intention, STmem_to_intention, Tmem_int):
        """
        Computes and returns intention activity, given memory activity, intention activity and their respective weights.
        """
        noise = np.random.uniform(-0.5, 0.5, 4)
        Wintention = np.multiply(LTintention, STintention)
        Wmem_to_int = np.multiply(LTmem_to_intention, STmem_to_intention)

        intention_activity = sigmoid(
            (np.dot(Wintention, intention_activity)+
            np.dot(Wmem_to_int, memory_activity)) - Tmem_int + noise
        )

        return intention_activity

    def confidence_activation(error_activity, STerr_to_conf, LTerr_to_conf):
        """
        Computes and returns confidence activity which is the opposite of error activity.
        It is highly negative when error_activity is near 1 and positive and near to 1 when error_activity is near 0.
        """
        return 1 - (error_activity*STerr_to_conf*LTerr_to_conf)


    def  go_activation(go_activity, reflexion_activity, STref_to_go, LTref_to_go, inhibition_activity, LTinhib_to_go, STinhib_to_go, Tgo):
        """
        Computes and return go activity given reflexion and inhibition activity.
        The go unit is not self-activated, it is only influenced by other units.
        """
        noise = np.random.uniform(-0.5, 0.5, 1)
        Wref_to_go = np.multiply(LTref_to_go, STref_to_go)
        Winhib_to_go = np.multiply(LTinhib_to_go, STinhib_to_go)

        go_activity = sigmoid(
            (#strong autoactivation gives bad results
            np.dot(Wref_to_go, reflexion_activity)+
            inhibition_activity*Winhib_to_go) - Tgo + noise)

        return go_activity

    def output_activation(output_activity,  LToutput, SToutput,intention_activity, LTintention_to_output, STintention_to_output, Toutput):
        """
        Computes and returns output activity, given intention activity, output activity and their respective weights.
        """   
        noise = np.random.uniform(-0.5, 0.5, 4)
        Wint_to_output = np.multiply(LTintention_to_output, STintention_to_output)
        Wout = np.multiply(LToutput, SToutput)

        output_activity = sigmoid(
            (np.dot(Wint_to_output, intention_activity)+
            np.dot(Wout, output_activity)) - Toutput + noise
        )

        return output_activity

    def error_activation(r, error_activity, intention_activity, LTerror, STerror, LTintention_to_error, STintention_to_error, Terror):
        """
        Computes and returns error activity, given intention activity, error activity, reward and their respective weights.
        """  
        noise = np.random.uniform(-0.5, 0.5, 1)
        Werror = np.multiply(LTerror, STerror)
        Wint_to_err = np.multiply(LTintention_to_error, STintention_to_error)

        error_activity = sigmoid(
            (  np.dot(Werror, r)+     
            np.dot(Werror, error_activity)+  
            np.dot(Wint_to_err, intention_activity))- Terror + noise
        ) 
        
        return error_activity

    def rules_activation(rules_activity, LTrules, STrules, Trule):
        """
        Computes and returns rules activity, given rules activity and their weights.
        """   
        noise = np.random.uniform(-0.5, 0.5, 3)
        Wrules = np.multiply(LTrules, STrules)

        rules_activity = sigmoid(
            np.dot(Wrules, rules_activity) - Trule + noise
        )

        return rules_activity

    def sigmoid(Z):
        return 1/(1+np.exp(-Z))

    def input_to_memSTupdt(STinput_to_mem, reflexion_activity):
        """
        Updates and returns input-to-memory Short-Term components, modulated by reflexion activity.
        Stops the entry of a new card into memory while reasoning on previous card needed.
        """     
        alpha = 0.4

        for i in range(0, STinput_to_mem.shape[0]):
            for j in range(0, STinput_to_mem.shape[0]):
                #if reflexion is activated, new card can be copied into memory
                if i==j and reflexion_activity > 0.5:
                    STinput_to_mem[i][j] = alpha * STinput_to_mem[i][j] + 1 - alpha
                if i==j and reflexion_activity < 0.5:
                    STinput_to_mem[i][j] = alpha * STinput_to_mem[i][j]
        
        return STinput_to_mem

    def int_to_mem_STupdt(STmem_to_intention, rules_activity):
        """
        Updates and returns intention-to-memory Short-Term components for all the 3 memory clusters assemblies, modulated by rule activities.
        """
        alpha = 0.4

        for i in range(0, 4):
            for j in range(0, 4):
                if (i-j)%4 == 0:
                    if rules_activity[0] > 0.5:
                        STmem_to_intention[i][j] = alpha * STmem_to_intention[i][j] + 1 - alpha 
                    elif rules_activity[0] < 0.5:
                        STmem_to_intention[i][j] = alpha * STmem_to_intention[i][j] 

        for i in range(0, 4):
            for j in range(4, 8):
                if (i-j)%4 == 0:
                    if rules_activity[1] > 0.5:
                        STmem_to_intention[i][i+4] = alpha * STmem_to_intention[i][i+4] + 1 - alpha 
                    elif rules_activity[1] < 0.5:
                        STmem_to_intention[i][i+4] = alpha * STmem_to_intention[i][i+4] 

        for i in range(0, 4):
            for j in range(8, 12):
                if (i-j)%4 == 0:
                    if rules_activity[2] > 0.5:
                        STmem_to_intention[i][i+8] = alpha * STmem_to_intention[i][i+8] + 1 - alpha
                    elif rules_activity[2] < 0.5:
                        STmem_to_intention[i][i+8] = alpha * STmem_to_intention[i][i+8]

        return STmem_to_intention

    def int_to_mem_LTupdt(LTmem_to_intention, STmem_to_intention, memory_activity, intention_activity, error_activity):
        """
        Updates and returns intention-to-memory Long-Term component for all the 3 memory clusters assemblies.
        Hebbian rule depending on error activity.
        """
        beta = 0.4

        for i in range(0, 4):
            for j in range(0, 12):
                if (i-j)%4 == 0:
                    LTmem_to_intention[i][j] = LTmem_to_intention[i][j] - beta * error_activity * STmem_to_intention[i][j] * memory_activity[j] * (2*intention_activity[i] - 1)

        return LTmem_to_intention

    def int_to_out_STupdt(STintention_to_output, go_activity):
        """
        Updates and returns intention-to-output Short-Term component, modulated by go unit and by hesitation about intention.
        """
        alpha = 0.4

        hesitation = 0
        for int_act in intention_activity:
            if int_act>0.02:
                hesitation+=1

        if go_activity < 0.5 or hesitation>=2:
            STintention_to_output = alpha * STintention_to_output
        if go_activity >= 0.5:
            STintention_to_output = alpha * STintention_to_output + 1 - alpha
        
        return STintention_to_output

    def int_to_err_STupdt(STintention_to_error, error_activity, intention_activity):
        """
        Updates and returns intention-to-error Short-Term component according to error and intention activities.
        """
        delta = 0.97
    
        for i in range(0, len(STintention_to_error)):
            if error_activity>0.5 and intention_activity[i] > 0.5:
                STintention_to_error[i] = delta * STintention_to_error[i] + 1 - delta
            else:
                STintention_to_error[i] = delta * STintention_to_error[i]
        
        return STintention_to_error

    def rules_auto_updt(STrules, error_activity, rules_activity):
        """
        Updates and returns rules auto-excitatory Short-Term components according to error and rule activities.
        Becomes a generator of diversity when error_activity and a rule activity are high enough.
        """
        recovery = 0.99 #memory of rejected rules
        delta = 0.97

        for i in range(0, STrules.shape[0]):
            Q = (error_activity*rules_activity[i])**2
            STrules[i][i] = (recovery*STrules[i][i] + (1 - recovery))* (1-Q) + delta*STrules[i][i]*Q
        return STrules

    def output_desactivation(SToutput, output_activity):
        """
        Updates and returns maximum output short-term component according to output activity itself.
        When the output is over 0.5, an action has been done and auto-activation is then decreased to avoid rapid constant neuron fire.
        """
        alpha = 0.4 #low alpha to avoid not to take action

        action = np.argmax(output_activity)
        if output_activity[action] >= 0.5:
            SToutput[action][action] = alpha * SToutput[action][action]
        if output_activity[action] < 0.5:
            SToutput[action][action] = alpha * SToutput[action][action] + 1 - alpha
        
        return SToutput

    def inhibition_desactivation(STinhib, confidence_activity):
        """
        Updates and returns inhibition long term component according to confidence activity.
        When confident about the action to take, inhibition is decreased to increase reflexion and go activities.
        """
        alpha = 0.4

        if confidence_activity >= 0.5:
            STinhib = alpha * STinhib
        if confidence_activity < 0.5:
            STinhib = alpha * STinhib + 1 - alpha
        
        return STinhib

    def external_feedback(action, response_card, reference_cards, rule):
        """
        Returns a true reward according to the success or not of the card chosen.
        """
        right_action_i = 0
        for i in range(0, nb_templates):
            if np.array_equal(reference_cards[i][rule], response_card[rule]):
                right_action_i = i

        if right_action_i == action:
            #0 to decrease error activity
            return 0
        else: 
            #1 to activate error cluster
            return 1

    def fake_reward(prev_reward, prev_chosen_int, intention_activity):
        """
        Returns a fake reward generated during internal auto-evaluation loop when reasoning.
        """
        current_int = np.argmax(intention_activity)
 
        if intention_activity[current_int] < 0.5:
            #No clear intention
            #0.40 to maintain generator of diversity without increasing it too much
            return 0.40

        if intention_activity[current_int] > 0.5:
            if prev_reward == 1:
                if current_int == prev_chosen_int:
                    #same error with new rule
                    return 1
                if current_int != prev_chosen_int:
                    #different action with new rule
                    return 0
            if prev_reward == 0:
                if current_int == prev_chosen_int:
                    #same success with new rule
                    return 0
                if current_int != prev_chosen_int:
                    #different action than the right one
                    return 1

    def rule_switching(rule):
        """
        Serially changing the rules : color - form - number.
        """
        if rule!=2:
            rule = rule+1
        else:
            rule = 0
        return rule

    def rule_history(r1,r2,r3,rule):
        if rule==0:
            r1.append(1)
            r2.append(0)
            r3.append(0)
        if rule==1:
            r2.append(1)
            r1.append(0)
            r3.append(0)
        if rule==2:
            r3.append(1)
            r1.append(0)
            r2.append(0)

        return r1, r2, r3

    ''' ----------------------------MAIN ---------------------------------- '''
    start = time.time()

    #WCST parameters
    nb_dim = 3
    nb_features = 4
    nb_templates = nb_features
    r = 3 #rules number

    #ACTIVITIES INITIALISATION
    memory_activity = np.zeros(nb_dim*nb_features, dtype="float")
    intention_activity = np.zeros(nb_templates, dtype="float")
    output_activity = np.zeros(nb_templates, dtype="float")
    error_activity = 0.
    rules_activity = np.zeros(3, dtype="float")
    go_activity = 0.
    confidence_activity = 0.
    reflexion_activity = 0.
    inhibition_activity = 0.

    #LONG TERM COMPONENTS INITIALISATION
    LTinput_to_mem = np.eye(4,4)*3
    LTmemory = np.full((4,4), -2, dtype="float") + np.eye(4,4)*8 #6 in diag, -2 out

    mem_to_int = np.eye(4,4)*3
    LTmem_to_intention = np.concatenate((mem_to_int,mem_to_int),axis=1)
    LTmem_to_intention = np.concatenate((LTmem_to_intention,mem_to_int),axis=1)
    LTintention = np.full((4,4), -2) + np.eye(4,4)*8

    LTinhib_to_go = -6
    LTref_to_go = 3
    LTintention_to_output = np.eye(4,4)*3
    LToutput = np.full((4,4), -2) + np.eye(4,4)*8
    LTout_to_inhib = np.full(4, 6)
    LTinhib = 6

    LTintention_to_error = np.full(4,5)#0 to lesion auto-evaluation
    LTerror = 6 #3 to lesion reward else 6
    LTerr_to_conf = 4

    LTconf_to_ref = 3
    LTinhib_to_ref = -6
    LTref = 6

    LTrules = np.full((3,3), -2) + np.eye(3,3)*8
    
    #SHORT TERM COMPONENTS INITIALISATION
    STinput_to_mem = np.eye(4, 4)
    STmemory = np.full((4,4), 1)

    STmem_to_intention = np.zeros((4,12))
    #STmem_to_intention = np.full((4,12),0.5) #to lesion rule coding network
    STintention = np.full((4,4), 1)

    STinhib_to_go = 1
    STref_to_go = 1
    STintention_to_output = np.zeros((4,4))
    SToutput = np.full((4,4), 1, dtype="float")
    STout_to_inhib = 1
    STinhib = 1
    
    STintention_to_error = np.zeros(4, dtype="float")
    STerror = 1
    STerr_to_conf = 1

    STconf_to_ref = 1
    STrule_to_ref = np.full((1,3), 1)
    STinhib_to_ref = 1
    STref = 1
    
    STrules = np.full((3,3), 1, dtype="float") - np.eye(3,3)

    #THRESHOLDS OF ACTIVATION
    Tmem_int = np.full(4, 3)
    Toutput = np.full(4, 4)
    Trule = np.full(3, 2)
    Terror = 5.5
    Tgo = 3
    Tref = 5
    Tinhib = 4

    #Data history for graphs
    mem_act_history = [] 
    r1_act_history = []
    r2_act_history = []
    r3_act_history = []
    r1 = []
    r2 = []
    r3 = []
    err_act_history = []
    go_his = []
    m = []
    o = []
    inhib = []
    ref = []
    int_act =[]
    input_list = []
    trial_his = []
    ptrial = []
    ntrial = []
    
    #INITIALISATION
    rule = 0
    nbTS = 0
    reward = 0
    prev_reward = 0
    trials = []
    t = 0
    loop = []
    l = 0
    rewards = []
    true_rewards = []
    nb_win = 0
    t_criterion = 0
    t_err = 0
    criterions = []
    prev_chosen_int = 0
    prev_chosen_rules = []
    nb_fake_reward = 0
    perseveration = 0
    winstreak = 0

    #Create reference cards
    m_percep = perception(nb_dim, nb_templates, nb_features)
    v_data = []

    #Picking a first random rule to follow
    rd = np.random.randint(0,2)
    rules_activity[rd] = 0.9
    STrules[rd] = 1

    #A card can be generated
    input_bool = True

    #START OF TEST
    while(nbTS<6):
        
        #Criterion test
        if winstreak==3:
            rule = rule_switching(rule)
            criterions.append(t_criterion)
            t_criterion = 0
            nbTS +=1
            winstreak = 0


        if (input_bool==True):
            #INPUT new card
            np_data, v_data = new_card()
            input_bool = False

        #ACTIVITIES COMPUTING 
            
        confidence_activity = confidence_activation(error_activity, STerr_to_conf, LTerr_to_conf)
        inhibition_activity = inhibition_activation(inhibition_activity, LTinhib, STinhib, output_activity, LTout_to_inhib, STout_to_inhib, Tinhib)
        reflexion_activity = reflexion_activation(reflexion_activity, LTref, STref, confidence_activity, STconf_to_ref, LTconf_to_ref, inhibition_activity, LTinhib_to_ref, STinhib_to_ref, Tref)
        memory_activity = memory_activation(memory_activity, v_data, LTinput_to_mem, LTmemory, STinput_to_mem, STmemory, Tmem_int) 
        rules_activity = rules_activation(rules_activity, LTrules, STrules, Trule)
        intention_activity = intention_activation(intention_activity, memory_activity, LTintention, STintention, LTmem_to_intention, STmem_to_intention, Tmem_int)
        go_activity =  go_activation(go_activity, reflexion_activity, STref_to_go, LTref_to_go, inhibition_activity, LTinhib_to_go, STinhib_to_go, Tgo)
        output_activity = output_activation(output_activity, LToutput, SToutput, intention_activity, LTintention_to_output, STintention_to_output, Toutput)


        if (output_activity[0]>0.5 or output_activity[1]>0.5 or output_activity[2]>0.5 or output_activity[3]>0.5):
            
            trials.append(t)
            trial_his.append(1)
            t+=1
            if t==62:
                milner_list = []#2nd deck of 64 cards for Milne

            prev_chosen_int = np.argmax(intention_activity)
            #ACTION AND FEEDBACK
            action = np.argmax(output_activity)
            reward = external_feedback(action, np_data, m_percep, rule)

            #DATA HISTORY
            rewards.append(reward)
            prev_reward = reward
            true_rewards.append(reward)

            #PERSEVERATION 
            prev_chosen_rules.append(np.argmax(rules_activity))
            chosen_rule = np.argmax(rules_activity)
            if t_err>=1 and reward==1:
                prev_rule = prev_chosen_rules[t-2]
                if prev_rule == chosen_rule:
                    perseveration += 1

            #print("\nACTION " + str(t)+ ": " + str(action)+ "  REWARD: " + str(reward))
            
            if reward == 0:
                t_err = 0
                nb_win += 1
                winstreak += 1
                ptrial.append(1)
                ntrial.append(0)

            if reward == 1:
                t_criterion += 1
                t_err += 1
                winstreak = 0
                ptrial.append(0)
                ntrial.append(1)

            input_bool =True
            #Clear output
        
        elif (reflexion_activity<0.5):
            #if reflexion phase on previous card, comparison can be done
            reward = fake_reward(prev_reward, prev_chosen_int, intention_activity)
            nb_fake_reward += 1
            if (t==0):
                reward = 0
            rewards.append(0)
        
        error_activity = error_activation(reward, error_activity, intention_activity, LTerror, STerror, LTintention_to_error, STintention_to_error, Terror)
        

        #WEIGHTS UPDATE
        #Short term and Long term updates
        STinput_to_mem = input_to_memSTupdt(STinput_to_mem, reflexion_activity)
        STmem_to_intention = int_to_mem_STupdt(STmem_to_intention, rules_activity) #comment to lesion rule clusters
        LTmem_to_intention = int_to_mem_LTupdt(LTmem_to_intention, STmem_to_intention, memory_activity, intention_activity, error_activity)
        STintention_to_output = int_to_out_STupdt(STintention_to_output, go_activity)
        STintention_to_error = int_to_err_STupdt(STintention_to_error, error_activity, intention_activity)
        STrules = rules_auto_updt(STrules, error_activity, rules_activity)    
        SToutput = output_desactivation(SToutput, output_activity)
        STinhib = inhibition_desactivation(STinhib, confidence_activity)


        #ACTIVITIES HISTORY for graph
        inhib.append(inhibition_activity)
        input_list.append(v_data)
        r1_act_history.append(rules_activity[0])
        r2_act_history.append(rules_activity[1])
        r3_act_history.append(rules_activity[2])
        m.append(memory_activity)
        int_act.append(intention_activity)
        o.append(output_activity)
        ref.append(reflexion_activity)
        err_act_history.append(error_activity)  
        go_his.append(go_activity)
        r1, r2, r3 = rule_history(r1, r2, r3, rule)

        l+=1
        loop.append(l)
        if input_bool!= True:
            trial_his.append(0)
            ptrial.append(0)
            ntrial.append(0)
        
        #In case the 6 criterias are not reached and no more cards
        if t==36:
            break
        #END OF NETWORK LOOP

    #Statistics
    if t>=1:


        err = 0
        for i in range(0, len(rewards)):
            if rewards[i] == 1:
                err +=1
        accuracy = nb_win / len(trials)
        #print("Accuracy : " + str(accuracy))

        t_crit_mean = np.mean(criterions)
        if len(criterions) == 0:
            t_crit_mean = err
        #print("Speed Learning: " + str(t_crit_mean))
    
        several_trials = 0
        single_trial = 0
        for i in range(0, len(true_rewards)-3):
            if true_rewards[i]==1 and true_rewards[i+1]==0 and true_rewards[i+2]==0 and true_rewards[i+3]==0:
                single_trial +=1
            if true_rewards[i]==1 and true_rewards[i+1]==1:
                several_trials += 1
    
        if single_trial==0 and several_trials==0:
            single_trial_lr = 0
        else:
            single_trial_lr = single_trial/(single_trial+several_trials)*100
        
        #print("Single trial learning : " + str(single_trial_lr))
        
        #print("Nb errors: " + str(err))
        persev_percent = perseveration/err*100
        #print("Percentage of Perseveration errors : " + str(persev_percent))
        #print("Task switching number : " + str(nbTS))
        #print("Number of trials : " + str(len(trials)))
        nb_trials = len(trials)
        #print("Number of loops : " + str(len(loop)))
    end = time.time()
    test_time = end-start
    #print("Temps d execution : %s secondes ---" % (end - start))
    
    #ACTIVITIES HISTORY PRINTING
    plt.subplot(4, 4, 1)
    plt.plot(loop, ptrial, "#75f33a", loop, ntrial, "#75063a" , loop, r1_act_history,  'b-' , loop, r2_act_history, 'g-', loop, r3_act_history,  'r-')
    plt.title("Rule activities")

    plt.subplot(4, 4, 2)
    plt.plot(loop, trial_his, "k:", loop, err_act_history)
    plt.title("Error activity ")

    plt.subplot(4, 4, 3)
    plt.plot(loop, trial_his, "k:", loop, r1, 'b-', loop, r2, 'g-', loop, r3, 'r-')
    plt.title("Hidden rules")

    plt.subplot(4, 4, 4)
    plt.plot(trials, true_rewards, 'r--')
    plt.title("True rewards")

    plt.subplot(4, 4, 5)
    plt.plot(loop, go_his)
    plt.title("Go activity")

    plt.subplot(4, 4, 6)
    plt.plot(loop, [item[0] for item in m], "r-", loop, [item[1] for item in m], "#166FB3" , loop, [item[2] for item in m], "#0BF320", loop, [item[3] for item in m], "#D22EC7")
    plt.title("Color memory")

    plt.subplot(4, 4, 7)
    plt.plot(loop, [item[4] for item in m], "r-", loop, [item[5] for item in m], "#166FB3" ,loop, [item[6] for item in m], "#0BF320" ,loop, [item[7] for item in m], "#D22EC7") 
    plt.title("Form memory")

    plt.subplot(4, 4, 8)
    plt.plot(loop, [item[8] for item in m],  "r-",loop, [item[9] for item in m], "#166FB3",loop, [item[10] for item in m],"#0BF320" ,loop, [item[11] for item in m], "#D22EC7")
    plt.title("Number memory")

    plt.subplot(4, 4, 9)
    plt.plot(loop, trial_his, "k:",loop, [item[0] for item in int_act], "r-", loop, [item[1] for item in int_act], "b-", loop, [item[2] for item in int_act], "g-", loop, [item[3] for item in int_act], "m-" )
    plt.title("Intentions")
    
    plt.subplot(4, 4, 10)
    plt.plot(loop, [item[0] for item in input_list], "r-", loop, [item[1] for item in input_list], "#166FB3", loop, [item[2] for item in input_list], "#0BF320", loop, [item[3] for item in input_list], "#D22EC7")
    plt.title("Input color")

    plt.subplot(4, 4, 11)
    plt.plot(loop, [item[4] for item in input_list], "r-", loop, [item[5] for item in input_list], "#166FB3",loop, [item[6] for item in input_list], "#0BF320",loop, [item[7] for item in input_list], "#D22EC7")
    plt.title("Input form")

    plt.subplot(4, 4, 12)
    plt.plot(loop, [item[8] for item in input_list], "r-",loop, [item[9] for item in input_list], "#166FB3",loop, [item[10] for item in input_list], "#0BF320",loop, [item[11] for item in input_list], "#D22EC7" )
    plt.title("Input number")

    plt.subplot(4, 4, 13)
    plt.plot(loop, trial_his, "k:", loop, [item[0] for item in o], "r-", loop, [item[1] for item in o], "b-", loop, [item[2] for item in o], "g-", loop, [item[3] for item in o], "m-" )
    plt.title("Output")

    plt.subplot(4, 4, 14)
    plt.plot(loop, trial_his, "k:",loop, ref)
    plt.title("Reflexion")

    plt.subplot(4, 4, 15)
    plt.plot(loop, trial_his, "k:",loop, inhib)
    plt.title("Inhibition")
    
    filename= str(path) + "/DehaeneNNPlot" + str(nb_test) + ".png"
    plt.savefig(filename)
    plt.close()
    
    #CLEAR COMPUTER MEMORY
    trials = None
    true_rewards = None
    rewards = None
    reasoning_list = None
    return nb_trials, t_crit_mean , single_trial_lr, persev_percent, nbTS, test_time

nb_test = 0
#save activities plot
path = "./activitiesPlot"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
nb_trials, t_crit_mean , single_trial_lr, persev_percent, nbTS, test_time = WCST_test(nb_test, path)
