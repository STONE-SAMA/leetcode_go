package leetcode

import (
	"math"
	"sort"
	"strconv"
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
	if len(nums) == 3 {
		if nums[0]+nums[1]+nums[2] == 0 {
			return [][]int{[]int{nums[0], nums[1], nums[2]}}
		}
		return [][]int{}
	}
	sort.Ints(nums)
	length := len(nums)
	res := make([][]int, 0)

	for left := 0; left < length-2; left++ {
		if left > 0 && nums[left-1] == nums[left] {
			continue
		}
		right := length - 1
		for middle := left + 1; middle < length-1; middle++ {
			if middle > left+1 && nums[middle-1] == nums[middle] {
				continue
			}
			for middle < right && nums[left]+nums[middle]+nums[right] > 0 {
				right--
			}
			if middle == right {
				break
			}
			if nums[left]+nums[middle]+nums[right] == 0 {
				res = append(res, []int{nums[left], nums[middle], nums[right]})
			}

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

func maxArea(height []int) int { //盛最多水的容器
	length := len(height)
	if length == 2 {
		return min(height[0], height[1])
	}
	temp := 0
	l, r := 0, length-1
	for l < r {
		temp = max(temp, min(height[l], height[r])*(r-l))
		if height[l] < height[r] {
			l++
		} else {
			r--
		}
	}
	return temp
}

func merge(intervals [][]int) [][]int { //合并区间
	if len(intervals) == 1 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	res := make([][]int, 0)
	left := intervals[0][0]
	right := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		temp_left := intervals[i][0]
		temp_right := intervals[i][1]
		if right < temp_left {
			res = append(res, []int{left, right})
			left = temp_left
			right = temp_right
		} else {
			right = max(temp_right, right)
		}
	}
	res = append(res, []int{left, right})
	return res
}

func DailyTemperatures(temperatures []int) []int { //每日温度
	if len(temperatures) == 1 {
		return []int{0}
	}
	res := make([]int, len(temperatures))
	var slices [][]int
	for index, value := range temperatures {
		if index == 0 {
			slices = append(slices, []int{index, value})
		} else {
			top := slices[len(slices)-1][1]
			if value > top {
				for value > top && len(slices) > 0 {
					temp_index := slices[len(slices)-1][0]
					days := index - temp_index
					res[temp_index] = days
					slices = slices[:len(slices)-1]
					if len(slices) > 0 {
						top = slices[len(slices)-1][1]
					}
				}
			}
			slices = append(slices, []int{index, value})
		}
	}
	return res
}

func Compress(chars []byte) int { //压缩字符串
	if len(chars) == 1 {
		return 1
	}
	pre := chars[0]
	mark := 1
	flag := 0
	for i := 1; i < len(chars); i++ {
		if chars[i] == pre {
			mark++
		} else {
			chars[flag] = pre
			flag++
			if mark > 1 { //整型拆分
				runes := []rune(strconv.Itoa(mark))
				for i := 0; i < len(runes); i++ {
					chars[flag] = byte(runes[i])
					flag++
				}
			}
			pre = chars[i]
			mark = 1
		}
	}
	//处理末位
	chars[flag] = pre
	flag++
	if mark > 1 {
		runes := []rune(strconv.Itoa(mark))
		for i := 0; i < len(runes); i++ {
			chars[flag] = byte(runes[i])
			flag++
		}
	}
	return flag
}

func IncreasingTriplet(nums []int) bool { //递增的三元子序列
	if len(nums) < 3 {
		return false
	}
	left := math.MaxInt32
	middle := math.MaxInt32
	for i := 0; i < len(nums); i++ {
		if nums[i] <= left {
			left = nums[i]
		} else if nums[i] <= middle {
			middle = nums[i]
		} else if nums[i] > middle {
			return true
		}
	}
	return false
}

func ProductExceptSelf(nums []int) []int { //除自身以外数组的乘积
	length := len(nums)
	left, right := make([]int, length), make([]int, length)
	left[0] = 1
	right[length-1] = 1
	for i := 1; i < length; i++ {
		left[i] = left[i-1] * nums[i-1]

	}
	for i := length - 2; i >= 0; i-- {
		right[i] = right[i+1] * nums[i+1]
	}
	res := make([]int, length)
	for i := 0; i < length; i++ {
		res[i] = left[i] * right[i]
	}
	return res
}

func closeStrings(word1 string, word2 string) bool {
	runes := []rune(word1)
	var cnt = make(map[rune]int)
	for i := 0; i < len(runes); i++ {
		cnt[runes[i]]++
	}
	runes = []rune(word2)
	var mark = make(map[rune]int)
	for i := 0; i < len(runes); i++ {
		if _, ok := cnt[runes[i]]; ok {
			mark[runes[i]]++
		} else {
			return false
		}
	}
	if len(mark) != len(cnt) {
		return false
	}
	var flag_num = make(map[int]int)
	for _, value := range cnt {
		flag_num[value]++
	}
	for _, value := range mark {
		if v, ok := flag_num[value]; !ok || v == 0 {
			return false
		} else {
			flag_num[value]--
		}
	}
	return true
}

func deleteMiddle1(head *ListNode) *ListNode { //删除链表的中间节点
	if head.Next == nil {
		return nil
	}
	node := head
	list := make([]*ListNode, 0)
	for node != nil {
		list = append(list, node)
		node = node.Next
	}
	mid := len(list) / 2
	pre := list[mid-1]
	if mid+1 < len(list) {
		pre.Next = list[mid+1]
	} else {
		pre.Next = nil
	}
	return head
}
func deleteMiddle2(head *ListNode) *ListNode { //删除链表的中间节点
	if head.Next == nil {
		return nil
	}
	slow, fast := head, head
	var prev *ListNode
	// 移动快慢指针
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	// 删除中间节点
	if prev != nil {
		prev.Next = slow.Next
	} else {
		// 如果链表只有两个节点，删除第二个节点
		head.Next = nil
	}
	return head
}

func OddEvenList(head *ListNode) *ListNode { //奇偶链表
	if head == nil { //0个节点
		return nil
	}
	if head.Next == nil || head.Next.Next == nil { //1、2个节点
		return head
	}
	oddNode := head
	evenHead := head.Next
	evenNode := evenHead
	for evenNode.Next != nil && evenNode.Next.Next != nil {
		oddNode.Next = evenNode.Next
		oddNode = oddNode.Next
		evenNode.Next = evenNode.Next.Next
		evenNode = evenNode.Next
	}
	if evenNode.Next != nil {
		oddNode.Next = evenNode.Next
		oddNode = oddNode.Next
	}
	oddNode.Next = evenHead
	evenNode.Next = nil
	return head
}

func pairSum(head *ListNode) int { //链表最大孪生和
	values := make([]int, 0)
	for head != nil {
		values = append(values, head.Val)
		head = head.Next
	}
	length := len(values)
	max_sum := 0
	for i := 0; i < length/2; i++ {
		max_sum = max(max_sum, values[i]+values[length-i-1])
	}
	return max_sum
}

func MaxOperations(nums []int, k int) int { // K 和数对的最大数目
	cnts := make(map[int]int)
	res := 0
	for i := 0; i < len(nums); i++ {
		val := nums[i]
		need := k - val
		if _, ok := cnts[need]; ok && cnts[need] > 0 {
			cnts[need]--
			res++
		} else {
			cnts[val]++
		}
	}
	return res
}

func removeStars(s string) string { //字符串移除*号
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if s[i] == '*' {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, s[i])
		}
	}
	return string(stack)
}

func asteroidCollision(asteroids []int) []int { //小行星碰撞
	stack := make([]int, 0)
	for i := 0; i < len(asteroids); i++ {
		if len(stack) > 0 {
			num := asteroids[i]
			flag := true //移动方向，true右，false左
			if num < 0 {
				flag = false
				num = -num
			}
			mark := true
			for mark {
				if len(stack) == 0 {
					stack = append(stack, asteroids[i])
					break
				}
				top := stack[len(stack)-1]
				if top > 0 && !flag { //栈顶向右，新元素向左
					if num == top {
						stack = stack[:len(stack)-1]
						mark = false
					} else if top < num {
						stack = stack[:len(stack)-1]
					} else {
						mark = false
					}
				} else {
					stack = append(stack, asteroids[i])
					mark = false
				}
			}
		} else {
			stack = append(stack, asteroids[i])
		}
	}
	return stack
}

func DecodeString(s string) string { //字符串解码
	stack := make([]rune, 0)
	runes := []rune(s)
	for i := 0; i < len(s); i++ {
		if runes[i] == ']' {
			temp := ""
			flag := true
			for flag {
				top := stack[len(stack)-1]
				if top != '[' {
					temp += string(top)
					stack = stack[:len(stack)-1]
				} else {
					stack = stack[:len(stack)-1] //左边的'['出
					flag = false
					nums := []int{}
					for len(stack) > 0 && stack[len(stack)-1] >= '0' && stack[len(stack)-1] <= '9' {
						nums = append(nums, int(stack[len(stack)-1]-'0'))
						stack = stack[:len(stack)-1]
					}
					num := 0
					for j := 0; j < len(nums); j++ {
						num += nums[j] * int(math.Pow(10, float64(j)))
					}
					str := ""
					temp = ReverseString(temp)
					for j := 0; j < int(num); j++ {
						str += temp
					}
					for x := 0; x < len(str); x++ {
						stack = append(stack, rune(str[x]))
					}
				}
			}
		} else {
			stack = append(stack, runes[i])
		}
	}
	return string(stack)
}
