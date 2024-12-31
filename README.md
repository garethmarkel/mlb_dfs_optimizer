# README

This repository implements the optimization procedure outlined in "How to Play Fantasy Sports Strategically (and Win)"
by Martin Haugh and Raghav Singal. It's implemented as a command line script--if you're interested in running this, it's
worth it to take the time to find or create high quality data sources. I may eventually add test files, however, from my own
pipeline. 

NOTE: Gurobi is not liscensced for commercial use. I implemented this as a technical project, but if you wish to actually bet on 
DFS with it you should use a different optimizer.

For now, the files you need to run this are:

1. A correlation matrix of expected fantasy point scoring for players in a given DFS contest. You can create this by estimating positional covariances and player level standard deviations from historical data, and combining them for a given pool of potential players.
2. A CSV with projected points for each player, team for each player, batting order for each player, and fantasy salary for each player
3. A matrix of simulated opponent lineups. The player distribution here shoudl as closely match what the actual draft proportions are, though this is obviously a challenging forecasting problem and beyond the scope of this repository.
4. A matrix of rder statistics and payouts (e.g. first place gets $100, second place gets $80, 3-5 get $60, etc.)

To pull up the helpfile, run `julia optimize_mlb_portfolio.jl --help`.