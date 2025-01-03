package leetcode

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func Max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func Min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func ReverseString(s string) string {
	runes := []rune(s)
	length := len(runes)
	for i := 0; i < length/2; i++ {
		runes[i], runes[length-1-i] = runes[length-1-i], runes[i]
	}
	return string(runes)
}
