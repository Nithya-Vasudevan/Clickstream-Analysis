# Clickstream-Analysis

Leverage features from clickstream
Actions, Intent, Behaviour and Outcomes; all four present highly correlated characteristics of a user on a Web Page. Actions of user on a system, though, can be representative of a certain intent. Ability to learn this intent through user's actions can help draw certain insight into the behaviour of users on a system.
The combined relation of these actions, intent and initial behaviour, leads to an outcome which can itself be either intermediate or final.
Analyse user's actions on an interactive system like a web page to learn their intent and do a prediction of the shopper's behaviour. Leverage features from clickstream data generated during live shopping sessions and broadly classify them into these groups


•	Users who are highly likely to make a purchase

•	Users who are unlikely to purchase and drop out from the path.

 



Task 1: We have given the a sequence of click events performed by some user during a typical session in the test file. The goal is to predict whether the user is going to buy something or not for the session.

Task 2: Answer the following questions based on the analysis 
1.	What are the key differentiating sequences/subsequence's/clicks between People ordered and Not ordered?
2.	What links or sequence of link lead to drop out after Adding to Cart and Checkout & where they drop out? (Checkout = 1 and order placed= 0)
3.	What links or sequence of link factors lead to customers not checking out at all (which is (Checkout = 0 and order placed= 0)
4.	How users in upper and Middle pushed into Lower, where will the intervention strategy for Middle and lower bucket happen for finishing order.

#Data Dictionary

ID		        : Session Id 

ad		        : ad clicked and then the Ad id if its blank no ads clicked

Link		        : what are the interaction clicks done in the Web

timestamp	: what time the click was done by a customer in the session .

checkout  	: if they added product and di checkout  0  No  1- Yes

order_placed	: Order placed after checkout  0 no  Yes - 1

grp		        : What group was the link  event part of  - (Grouping of all similar links that has same function)

funnel_level  	: what is the stage of the action click .


•	The file comprising the clicks of the users over a session and that session resulting on an order or a user just did a checkout or a user exited before a checkout.

•	Each record/line in the file is an action or a click done by a user in that session, each session is a unique user. They will have multiple records in a session based on the actions and clicks they are doing to complete an order or add to a checkout or browsing and exiting without checkout or order.

•	The Timestamp field when its sorted on the session give the sequence of the event or links by the customer in the session , there can be multiple clicks on the link and to and fro clicks that will be there as it's all part of user behaviour.

•	The funnel name will give info of how the customers traverse through the website , Customer start with upper funnel where they do learn and do product verification and then they go to middle where they select products and then finish up in lower with payment and other address info.

•	Target label for our dataset was the presence or absence of a conversion in the session which is Order placed 0 or 1.
