# Long_Haul_Freight
Groups lanes together.

Situation & Task:  As a data science consultant at a third-party logistics startup,  I created an application that clarified an annual bidding process. The business submitted annual bids for a few thousand routes spanning hundreds of major US cities. 

Action:  My application identified groupings of 10-20 routes that could be priced on the same level in our pricing model.
On my own time, I pushed the model to cluster five-thousand routes using the OPTICS algorithm, and I added functionalities that allow you to visualize the primary arteries in your logistics network.

The following application outputs improve the clarity and interpretability an annual bid process.
Sample routes (Sample_routes.html)  A map of the simple random sample of 200 / 5,000 routes                                                                
Create groups (Create_groups.html)  A map of the routes grouped together. Each line represents a group of 10-20 routes.         
Price zones (Price_zones.html)      A map of the price zones. Each price zone is a pair of shapes that share a color.
Identify hubs (Identify_hubs.html)  A map of the route groupings. Each hub is assigned a color.                                 
Rank hubs (Rank_hubs.html)          A map of hubs are connected by many routes flowing back and forth between hubs. The PageRank algorithm allows us to focus on the most important hubs. Larger circles represent higher importance. Thicker connections represent more routes.

Results:   
