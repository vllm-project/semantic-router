"""vLLM logo printing utilities."""

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_ORANGE = "\033[38;2;254;181;22m"  # #FEB516 - vLLM V left side
COLOR_BLUE = "\033[38;2;48;162;255m"  # #30A2FF - vLLM V right side
COLOR_WHITE = "\033[97m"  # White - for LLM and SR


def print_vllm_logo():
    """Print the vLLM SR logo with colors.

    Logo design: vLLM SR
    - v = left side orange, right side blue
    - LLM SR = white
    """
    logo = [
        "",
        f"{COLOR_ORANGE}##{COLOR_WHITE}          {COLOR_BLUE}##  {COLOR_WHITE}##        ##        ##      ##    ######    ########{COLOR_RESET}",
        f"{COLOR_ORANGE} ##{COLOR_WHITE}        {COLOR_BLUE}##   {COLOR_WHITE}##        ##        ###    ###   ##    ##   ##    ##{COLOR_RESET}",
        f"{COLOR_ORANGE}  ##{COLOR_WHITE}      {COLOR_BLUE}##    {COLOR_WHITE}##        ##        ####  ####   ##         ##    ##{COLOR_RESET}",
        f"{COLOR_ORANGE}   ##{COLOR_WHITE}    {COLOR_BLUE}##     {COLOR_WHITE}##        ##        ## #### ##    ####     ########{COLOR_RESET}",
        f"{COLOR_ORANGE}    ##{COLOR_WHITE}  {COLOR_BLUE}##      {COLOR_WHITE}##        ##        ##  ##  ##       ##    ##  ##{COLOR_RESET}",
        f"{COLOR_ORANGE}     ##{COLOR_BLUE}##       {COLOR_WHITE}##        ##        ##      ##   ##    ##   ##   ##{COLOR_RESET}",
        f"{COLOR_ORANGE}      {COLOR_BLUE}##        {COLOR_WHITE}########  ########  ##      ##    ######    ##    ##{COLOR_RESET}",
        "",
    ]

    for line in logo:
        print(line)
