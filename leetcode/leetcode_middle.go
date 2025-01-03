package leetcode

import (
	"math"
	"sort"
)

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var head *ListNode
	var flag *ListNode
	mark := 0
	for l1 != nil || l2 != nil {
		v1, v2 := 0, 0
		if l1 != nil {
			v1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			v2 = l2.Val
			l2 = l2.Next
		}
		sum := v1 + v2 + mark
		mark = sum / 10
		sum = sum % 10
		if head == nil {
			head = &ListNode{Val: sum}
			flag = head
		} else {
			flag.Next = &ListNode{Val: sum}
			flag = flag.Next
		}
	}
	if mark == 1 {
		flag.Next = &ListNode{Val: 1, Next: nil}
	}
	return head
}

func LengthOfLongestSubstring(s string) int {
	var cntMap = make(map[string]bool) //存储已有元素
	runes := []rune(s)
	res := 0
	for left, right := 0, 0; right < len(runes); right++ {
		temp := string(runes[right])
		_, ok := cntMap[temp]
		for ok {
			delete(cntMap, string(runes[left]))
			left++
			ok = cntMap[temp]
		}
		cntMap[temp] = true
		res = Max2(res, right-left+1)
	}
	return res
}

func reverse(x int) int {
	var rev int
	for x != 0 {
		if rev < math.MinInt32/10 || rev > math.MaxInt32/10 {
			return 0
		}
		digit := x % 10
		x /= 10
		rev = rev*10 + digit
	}
	return rev
}

func removeDuplicates1(nums []int) int {
	var conts = make(map[int]int)
	l, r := 0, 0
	k := 0
	for r < len(nums) {
		val, ok := conts[nums[r]]
		if ok { //已有元素
			if val == 1 {
				conts[nums[r]]++
				nums[l] = nums[r]
				l++
				k++
			} else {
				r++
			}
		} else {
			conts[nums[r]] = 1
			nums[l] = nums[r]
			r++
			l++
			k++
		}
	}
	return k
}

func majorityElement(nums []int) int {
	var cnts = make(map[int]int)
	length := len(nums)
	for _, num := range nums {
		_, ok := cnts[num]
		if ok {
			cnts[num]++
			if cnts[num] > length/2 {
				return num
			}
		} else {
			cnts[num] = 1
		}

	}
	return nums[0]
}

func Rotate(nums []int, k int) { //轮转数组
	length := len(nums)
	k %= length
	if k == 0 {
		return
	}
	temp := make([]int, length)
	copy(temp[k:], nums[:length-k])
	copy(temp[:k], nums[length-k:])
	for index, val := range temp {
		nums[index] = val
	}
	return
}

func maxProfit1(prices []int) int { //买卖股票的最佳时机
	min_value := math.MaxInt16
	max_value := 0
	for i := 0; i < len(prices); i++ {
		if min_value > prices[i] {
			min_value = prices[i]
		} else {
			max_value = Max2(max_value, prices[i]-min_value)
		}
	}
	return max_value
}

func maxProfit2(prices []int) int { //买卖股票的最佳时机 II
	if len(prices) == 1 {
		return 0
	}
	pre := prices[0]
	profit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > pre {
			profit += prices[i] - pre
		}
		pre = prices[i]
	}
	return profit
}

func canJump1(nums []int) bool {
	if len(nums) == 1 {
		return true
	}
	mark := nums[0]
	for i := 0; i < mark; i++ {
		mark = max(mark, nums[i]+i)
		if mark > len(nums)-1 {
			return true
		}
	}
	return false
}

func Jump2(nums []int) int { //跳跃游戏 II
	if len(nums) == 1 {
		return 0
	}
	steps := 0
	end := 0
	flag := 0 //当前位置
	for i := 0; i < len(nums)-1; i++ {
		flag = max(flag, nums[i]+i)
		if end == i {
			end = flag
			steps++
		}
	}
	return steps
}

func threeSum(nums []int) [][]int { //三数之和
	sort.Ints(nums)
	res := make([][]int, 0)
	l, m, r := 0, 0, len(nums)-1
	for l < r {
		left := nums[l]
		right := nums[r]
		m = l + 1
		for m < r {
			if left+right+nums[m] == 0 {
				res = append(res, []int{left, right, nums[m]})
			}
			m++
		}

	}
	return res
}

func twoSum2(numbers []int, target int) []int { //两数之和 II - 输入有序数组
	l := 0
	r := len(numbers) - 1
	for l < r {
		temp := numbers[l] + numbers[r]
		if temp == target && numbers[l] != numbers[r] {
			return []int{l + 1, r + 1}
		} else if temp > target {
			r--
		} else {
			l++
		}
	}
	return []int{}
}

//func maxArea(height []int) int { //盛最多水的容器
//
//}
