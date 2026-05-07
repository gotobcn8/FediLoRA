Missing rate: 
- 0.4 (priority)
- 0.5 or 0.3 (Unsured)

### recaps
each client has at least 10K size dataset, each time we select 3000+ to train, we abandon testset for each client
, and the recaps_testset will be 10K+
- Total 8 communication rounds (At least 5)
- client_selection_frac : 0.4
- training subrate: 0.3
- test_parts: 0.2 (which means 2000 data used to inference each time)


### sam
each client has at least 2K size dataset, each time we select around 1K to train, we abandon testset for each client
, and the {sam}_test will be 2K+
- Total 8 communication rounds (At least 5)
- client_selection_frac : 0.4
- training subrate: 0.5
- test_parts: 0.4 (which means 800 data used to inference each time)


### next
each client has at least 1.8K size dataset, each time we select around 1K to train, we abandon testset for each client
, and the {next}_test will be 2K+
- Total 8 communication rounds (At least 5)
- client_selection_frac : 0.4
- training subrate: 0.5
- test_parts: 3 (which means 800 data used to inference each time)