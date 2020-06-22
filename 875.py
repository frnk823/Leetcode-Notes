class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        # 暴力求解法，但是从平均数开始遍历，加速算法
        sum=0
        for i in piles:
            sum += i
        if sum%H == 0:
            avg = int(sum / H)
        else:
            avg = int(sum / H) + 1
        # 暴力遍历，如果avg满足题意，就返回，不满足就avg+=1继续尝试
        while(1):
            sum = 0
            for i in piles:
                if (i % avg == 0):
                    sum += int(i / avg)
                else:
                    sum += int(i / avg) + 1
            if(sum <= H):
                return avg
            avg += 1