Situation / Task:  As a data science consultant at a third-party logistics startup,
                   I created an application that simplified an annual bidding process.

Situation / Task:  The business submitted annual bids for 1-5 thousand routes spanning hundreds of major US cities.
    image           a_routes.html
    description     A simple random sample of 200 / 5,000 routes to simulate the typical customer.

Task:  To increase clarity, my application used a clustering algorithm to remove outliers and identify groupings of 10-20 routes.
    image           b_route_groups.html
    description     A map of the route groupings. Each line represents a grouping of 10-20 routes.

Action:  I added features on my own time, so the features onward were not productionalized.

Results:
    Pricing zones
        image               c_pricing_zones.html
        description         A map of the pricing zones. Each pricing zone is a pair of shapes that share a color.

    Hub identification
        image               d_hub_colors.html
        description         A map of the route groupings. Each hub is assigned a color.
        
    The hubs are connected by many routes flowing back and forth between the hubs.
    The PageRank algorithm allows us to focus on the most important hubs.
        image               e_hubs.html
        description         This is a network analysis of the routes and the hubs they connect.
                            Larger circles represent more important hubs.
                            The triangular connections represent routes flowing into the hub.
                            The thicker connections represent more routes.
                            
                            
                            