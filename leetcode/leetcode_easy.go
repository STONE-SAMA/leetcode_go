package leetcode

import (
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func twoSum(nums []int, target int) []int {
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i]+nums[j] == target {
				return []int{i, j}
			}
		}
	}
	return nil
}

func searchInsert(nums []int, target int) int {
	for index, value := range nums {
		if value >= target {
			return index
		}
	}
	return len(nums)
}

func lengthOfLastWord(s string) int {
	strs := strings.Fields(s)
	for i := len(strs); i > 0; i-- {
		if len(strs[i-1]) > 0 {
			return len(strs[i-1])
		}
	}
	return len(strs)
}

func PlusOne(digits []int) []int {
	length := len(digits)
	if digits[length-1] > 8 {
		length := length - 1
		digits[length] = 0
		flag := false
		for i := length - 1; i >= 0; i-- {
			if digits[i] == 9 {
				digits[i] = 0
				continue
			} else {
				digits[i] += 1
				flag = true
				break
			}
		}
		if flag {
			return digits
		} else {
			res := append([]int{1}, digits...)
			return res
		}

	} else {
		digits[length-1] += 1
		return digits
	}
}

func AddBinary(a string, b string) string {
	reverse_a := ReverseString(a)
	reverse_b := ReverseString(b)
	length := Max2(len(reverse_a), len(reverse_b))
	flag := 0
	var builder strings.Builder
	var res string
	for i := 0; i < length; i++ {
		first := 0
		second := 0
		if i < len(reverse_a) {
			first = int(reverse_a[i] - '0')
		}
		if i < len(reverse_b) {
			second = int(reverse_b[i] - '0')
		}
		flag = first + second + flag
		if flag > 1 {
			builder.WriteString(strconv.Itoa(flag - 2))
			flag = 1
		} else {
			builder.WriteString(strconv.Itoa(flag))
			flag = 0
		}
	}
	if flag == 1 {
		builder.WriteString(strconv.Itoa(flag))
	}
	res = ReverseString(builder.String())
	return res
}

func climbStairs(n int) int {
	a := 1 //爬一层楼
	b := 2 //爬两层楼
	if n == 1 {
		return a
	} else if n == 2 {
		return b
	} else {
		r := 0
		for i := 3; i < n; i++ {
			r = a + b
			b = a
			a = r
		}
		return r
	}
}

func climbStairs_dp(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	flag := head.Val
	var pre *ListNode
	node := head
	pre = node
	for head.Next != nil {
		head = head.Next
		if head.Val != flag {
			node.Next = head
			node = head
			flag = head.Val
		}
	}
	node.Next = nil
	return pre
}

func MergeNums(nums1 []int, m int, nums2 []int, n int) {
	nums_temp := make([]int, m)
	for i := 0; i < m; i++ {
		nums_temp[i] = nums1[i]
	}
	i, j := 0, 0
	index := 0
	for i < m && j < n {
		if nums_temp[i] <= nums2[j] {
			nums1[index] = nums_temp[i]
			i++
		} else {
			nums1[index] = nums2[j]
			j++
		}
		index++
	}
	for i < m {
		nums1[index] = nums_temp[i]
		i++
		index++
	}
	for j < n {
		nums1[index] = nums2[j]
		j++
		index++
	}
}

func inorderTraversal(root *TreeNode) []int {
	var num int
	if root == nil {
		return nil
	} else {
		num = root.Val
	}

	var nums1, nums2 []int
	if root.Left != nil {
		nums1 = inorderTraversal(root.Left)
	}
	if root.Right != nil {
		nums2 = inorderTraversal(root.Right)
	}
	slice1 := nums1
	slice1 = append(slice1, num)
	slice2 := nums2
	result := append(slice1, slice2...)
	return result
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	} else if p == nil || q == nil {
		return false
	} else if p.Val != q.Val {
		return false
	} else {
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return compareTreeNodes(root.Left, root.Right)
}

func compareTreeNodes(left *TreeNode, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	} else if left == nil || right == nil || left.Val != right.Val {
		return false
	}
	return compareTreeNodes(left.Left, right.Right) && compareTreeNodes(left.Right, right.Left)
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	var depth int
	left := maxDepth(root.Left)
	right := maxDepth(root.Right)
	depth = max(left, right) + 1
	return depth
}

func sortedArrayToBST(nums []int) *TreeNode {
	return createDFS(nums, 0, len(nums)-1)
}

func createDFS(nums []int, left int, right int) *TreeNode {
	if left > right {
		return nil
	}
	mid := (left + right) / 2
	root := &TreeNode{Val: nums[mid]}
	root.Left = createDFS(nums, left, mid-1)
	root.Right = createDFS(nums, mid+1, right)
	return root
}

func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if math.Abs(float64(compareLeftRight(root.Left)-compareLeftRight(root.Right))) > 1 {
		return false
	}
	return isBalanced(root.Left) && isBalanced(root.Right)
}

func compareLeftRight(node *TreeNode) int {
	if node == nil {
		return 0
	}
	left := compareLeftRight(node.Left)
	right := compareLeftRight(node.Right)
	return max(left, right) + 1
}

func isPalindrome(s string) bool {
	s = strings.ToLower(s)
	runes := []rune(s)
	charrunes := make([]rune, len(runes))
	index := 0
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		if c >= '0' && c <= '9' {
			charrunes[index] = c
			index++
		}
		if c >= 'a' && c <= 'z' {
			charrunes[index] = c
			index++
		}
	}
	for i := 0; i < index/2; i++ {
		if charrunes[i] != charrunes[index-i-1] {
			return false
		}
	}
	return true
}

func singleNumber(nums []int) int {
	result := 0
	for _, num := range nums {
		result ^= num
	}
	return result
}

func hasCycle(head *ListNode) bool {
	visited := map[*ListNode]struct{}{}
	for head != nil {
		if _, ok := visited[head]; ok {
			return true
		}
		visited[head] = struct{}{}
		head = head.Next
	}
	return false
}

func ThirdMax(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	if len(nums) == 2 {
		if nums[0] > nums[1] {
			return nums[1]
		}
		if nums[1] > nums[0] {
			return nums[0]
		}
	}
	a, b, c := math.MinInt64, math.MinInt64, math.MinInt64
	for _, num := range nums {
		if a == num || b == num || c == num {
			continue
		}
		if num >= a {
			c = b
			b = a
			a = num
		} else if num >= b {
			c = b
			b = num
		} else if num >= c {
			c = num
		}
	}
	if c != math.MinInt64 {
		return c
	}
	return a
}

func AddStrings(num1 string, num2 string) string {
	num1 = ReverseString(num1)
	num2 = ReverseString(num2)
	len1 := len(num1)
	len2 := len(num2)
	length := Min2(len1, len2)
	var builder strings.Builder
	flag := 0
	i := 0
	for i = 0; i < length; i++ {
		temp := int(num1[i]-'0') + int(num2[i]-'0') + flag
		if temp > 9 {
			temp = temp % 10
			flag = 1
		} else {
			flag = 0
		}
		builder.WriteString(strconv.Itoa(temp))
	}
	for i < len1 {
		temp := int(num1[i]-'0') + flag
		if temp > 9 {
			temp = temp % 10
			flag = 1
		} else {
			flag = 0
		}
		builder.WriteString(strconv.Itoa(temp))
		i++
	}
	for i < len2 {
		temp := int(num2[i]-'0') + flag
		if temp > 9 {
			temp = temp % 10
			flag = 1
		} else {
			flag = 0
		}
		builder.WriteString(strconv.Itoa(temp))
		i++
	}
	if flag == 1 {
		builder.WriteString("1")
	}
	return ReverseString(builder.String())
}

func countSegments(s string) int {
	if len(s) == 0 {
		return 0
	}
	result := strings.Split(s, " ")
	count := 0
	for _, str := range result {
		if str != "" {
			count++
		}
	}
	return count
}

func findDisappearedNumbers(nums []int) []int {
	length := len(nums)
	flags := make([]int, length)
	for i := 0; i < length; i++ {
		index := nums[i]
		flags[index-1]++
	}
	var result []int
	for i := 0; i < length; i++ {
		if flags[i] == 0 {
			result = append(result, i+1)
		}
	}
	return result
}

func findMaxConsecutiveOnes(nums []int) int {
	max_num := 0
	count := 0
	for _, value := range nums {
		if value == 1 {
			count++
		} else {
			max_num = Max2(max_num, count)
			count = 0
		}
	}
	max_num = Max2(max_num, count)
	return max_num
}

func firstUniqChar(s string) int {
	count := make([]int, 26)
	for _, ch := range s {
		count[ch-'a']++
	}
	for index, val := range s {
		if count[val-'a'] == 1 {
			return index
		}
	}
	return -1
}

func findTheDifference(s string, t string) byte {
	count_s := make([]int, 26)
	for _, ch := range s {
		count_s[ch-'a']++
	}
	for _, ch := range t {
		count_s[ch-'a']--
	}
	for index, val := range count_s {
		if val == -1 {
			return byte(index + 'a')
		}
	}
	return 0
}

func backspaceCompare(s string, t string) bool {
	var slices []string
	for _, v := range s {
		if v != '#' {
			slices = append(slices, string(v))
		} else {
			if len(slices) == 0 {
				continue
			}
			slices = slices[:len(slices)-1]
		}
	}
	var slice_t []string
	for _, v := range t {
		if v != '#' {
			slice_t = append(slice_t, string(v))
		} else {
			if len(slice_t) == 0 {
				continue
			}
			slice_t = slice_t[:len(slice_t)-1]
		}
	}
	if len(slice_t) == len(slices) {
		for i := 0; i < len(slices); i++ {
			if slices[i] != slice_t[i] {
				return false
			}
		}
		return true
	}
	return false
}

func middleNode(head *ListNode) *ListNode {
	var nodes []*ListNode
	for head != nil {
		nodes = append(nodes, head)
		head = head.Next
	}
	length := len(nodes)
	middle := nodes[length/2]
	return middle
}

func largestSumAfterKNegations(nums []int, k int) int {
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] < 0 && k > 0 {
			nums[i] = -nums[i]
			k--
		}
	}
	for i := 0; i < k; i++ {
		sort.Ints(nums)
		nums[0] = -nums[0]
	}
	sum := 0
	for _, num := range nums {
		sum += num
	}
	return sum
}

func RemoveDuplicates(s string) string {
	if len(s) == 1 {
		return s
	}
	var slices []string
	for _, ch := range s {
		if len(slices) > 0 && slices[len(slices)-1] == string(ch) {
			slices = slices[:len(slices)-1]
		} else {
			slices = append(slices, string(ch))
		}
	}
	return strings.Join(slices, "")
}

func ToGoatLatin(sentence string) string {
	sentences := strings.Split(sentence, " ")
	var res string
	for index, str := range sentences {
		runes := []rune(str)
		if unicode.ToLower(runes[0]) != 'a' && unicode.ToLower(runes[0]) != 'e' && unicode.
			ToLower(runes[0]) != 'i' && unicode.ToLower(runes[0]) != 'o' && unicode.ToLower(runes[0]) != 'u' {
			temp := runes[0]
			runes = runes[1:]
			runes = append(runes, temp)
		}
		runes = append(runes, 'm')
		runes = append(runes, 'a')
		for i := 0; i <= index; i++ {
			runes = append(runes, 'a')
		}
		if index < len(sentences)-1 {
			runes = append(runes, ' ')
		}
		res += string(runes)
	}
	return res
}

func RelativeSortArray(arr1 []int, arr2 []int) []int {
	cnts := make([][2]int, len(arr2))
	for i := 0; i < len(arr2); i++ {
		cnts[i][0] = arr2[i]
	}
	index := len(cnts)
	temp := 0
	for _, value := range arr1 {
		flag := false
		for i := 0; i < index; i++ {
			if value == cnts[i][0] {
				cnts[i][1]++
				flag = true
				if i < len(arr2) {
					temp++
				}
			}
		}
		if !flag {
			index++
			cnts = append(cnts, [2]int{value, 1})
		}
	}
	var res []int
	for _, value := range cnts {
		val := value[0]
		cnt := value[1]
		for i := 0; i < cnt; i++ {
			res = append(res, val)
		}
	}
	sort.Slice(res[temp:], func(i, j int) bool {
		return res[temp+i] < res[temp+j]
	})
	return res
}

func removeElement(nums []int, val int) int {
	var k int
	l := 0
	r := len(nums) - 1
	for l < r {
		if nums[l] == val {
			r--
			for nums[r] == val && r > l {
				r--
			}
			nums[l] = nums[r]
		} else {
			l++
			k++
		}
	}
	return k
}

func RemoveDuplicates_int(nums []int) int {
	var cnts = make(map[int]bool)
	k := 0
	l := 0
	r := 0
	for r < len(nums) {
		_, ok := cnts[nums[r]]
		if ok {
			r++
		} else {
			cnts[nums[r]] = true
			nums[l] = nums[r]
			l++
			r++
			k++
		}
	}
	return k
}

func moveZeroes(nums []int) { //移动零
	left, right := 0, 0
	for right < len(nums) {
		if nums[right] == 0 {
			right++
		} else {
			nums[left] = nums[right]
			left++
			right++
		}
	}
	for left < len(nums) {
		nums[left] = 0
		left++
	}
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	nodes := make(map[*ListNode]bool)
	for headA != nil {
		nodes[headA] = true
		headA = headA.Next
	}
	for headB != nil {
		if _, ok := nodes[headB]; ok {
			return headB
		}
		headB = headB.Next
	}
	return nil
}

func isPalindrome2(head *ListNode) bool {
	nums := []int{}
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}
	length := len(nums)
	for i := 0; i < length/2; i++ {
		if nums[i] != nums[length-i-1] {
			return false
		}
	}
	return true
}

func mergeAlternately(word1 string, word2 string) string { //交替合并字符串
	runes1 := []rune(word1)
	runes2 := []rune(word2)
	length := min(len(runes1), len(runes2))
	var res []rune
	for i := 0; i < length; i++ {
		res = append(res, runes1[i])
		res = append(res, runes2[i])
	}
	var temp string
	if len(word1) > length {
		temp = string(runes1[length:])
	}
	if len(word2) > length {
		temp = string(runes2[length:])
	}
	return string(res) + temp
}

func kidsWithCandies(candies []int, extraCandies int) []bool { //拥有最多糖果的孩子
	num_max := -1
	for i := 0; i < len(candies); i++ {
		num_max = max(num_max, candies[i])
	}
	res := make([]bool, len(candies))
	for i := 0; i < len(candies); i++ {
		if candies[i]-num_max+extraCandies >= 0 {
			res[i] = true
		}
	}
	return res
}

func reverseVowels(s string) string {
	var indexs []int
	var values []rune
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		temp := runes[i]
		temp = unicode.ToLower(temp)
		if temp == 'a' || temp == 'e' || temp == 'i' || temp == 'o' || temp == 'u' {
			indexs = append(indexs, i)
			values = append(values, runes[i])
		}
	}
	for i := 0; i < len(indexs); i++ {
		runes[indexs[i]] = values[len(values)-i-1]
	}
	return string(runes)
}

func findDifference(nums1 []int, nums2 []int) [][]int { //找出两数组的不同
	var cnt_nums1 = make(map[int]bool)
	var cnt_nums2 = make(map[int]bool)
	var res = make([][]int, 2)
	for i := 0; i < len(nums1); i++ {
		cnt_nums1[nums1[i]] = true
	}
	for i := 0; i < len(nums2); i++ {
		cnt_nums2[nums2[i]] = true
		if _, ok := cnt_nums1[nums2[i]]; !ok {
			res[1] = append(res[1], nums2[i])
			cnt_nums1[nums2[i]] = true
		}
	}
	for i := 0; i < len(nums1); i++ {
		if _, ok := cnt_nums2[nums1[i]]; !ok {
			res[0] = append(res[0], nums1[i])
			cnt_nums2[nums1[i]] = true
		}
	}
	return res
}

func UniqueOccurrences(arr []int) bool { //独一无二的出现次数
	var cnt = make(map[int]int)
	for _, value := range arr {
		cnt[value]++
	}
	var flag = make(map[int]bool)
	for _, v := range cnt {
		if _, ok := flag[v]; ok {
			return false
		} else {
			flag[v] = true
		}
	}
	return true
}

func reverseList(head *ListNode) *ListNode { //反转链表
	if head == nil {
		return nil
	}
	if head.Next == nil {
		return head
	}
	node := head
	var pre *ListNode = nil
	for node.Next != nil {
		temp := node.Next
		node.Next = pre
		pre = node
		node = temp
	}
	node.Next = pre
	return node
}

func FindMaxAverage(nums []int, k int) float64 { //子数组最大平均数 I
	MaxSum := 0
	for i := 0; i < k; i++ {
		MaxSum += nums[i]
	}
	if len(nums) == k {
		return float64(MaxSum) / float64(k)
	} else {
		mark := MaxSum
		for i := 1; i <= len(nums)-k; i++ {
			temp := nums[i+k-1] + mark - nums[i-1]
			mark = temp
			MaxSum = max(temp, MaxSum)
		}
	}
	return float64(MaxSum) / float64(k)
}
