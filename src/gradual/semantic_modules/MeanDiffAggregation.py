class MeanDiffAggregation:
    def __init__(self) -> None:
        self.name = "MeanDiffAggregation"
        pass

    def aggregate_strength(self, attackers, supporters):
        sum_attacker = 0
        num_attacker = 0
        sum_supporter = 0
        num_supporter = 0

        for a in attackers:
            num_attacker += 1
            sum_attacker += a

        if num_attacker == 0:
            mean_attacker = 0
        else:
            mean_attacker = sum_attacker / num_attacker
        
        for s in supporters:
            num_supporter += 1
            sum_supporter += s

        if num_supporter == 0:
            mean_supporter = 0
        else:
            mean_supporter = sum_supporter / num_supporter

        return mean_supporter - mean_attacker

    def __str__(self) -> str:
        return __class__.__name__
