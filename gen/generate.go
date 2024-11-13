// Package gen generates Go code from an XGBoost model.
package gen

import (
	"fmt"
	"go/format"
	"os"
)

type treeFunction struct {
	Code  string
	Name  string
	Class int
}

func generateSource(
	packageName,
	funcName string,
	baseScore float32,
	numClasses int64,
	trees []*node,
	r *renderer,
) (string, error) {
	var treeFunctions []treeFunction
	for i, t := range trees {
		level := 0
		code, err := codegenTree(r, t, level)
		if err != nil {
			return "", err
		}

		treeFunctions = append(
			treeFunctions,
			treeFunction{
				Code:  code,
				Name:  fmt.Sprintf("tree%s_%d", funcName, i),
				Class: t.data.Class,
			},
		)
	}

	code, err := r.executeRoot(packageName, funcName, baseScore, numClasses, treeFunctions)
	if err != nil {
		return "", err
	}

	// We run the code through formatting to check for syntax errors. We don't
	// return the formatted code since we intend what we generate to already be
	// well formatted.
	if _, err := format.Source([]byte(code)); err != nil {
		return "", fmt.Errorf("error formatting code: %w", err)
	}

	return code, nil
}

func codegenTree(r *renderer, tree *node, level int) (string, error) {
	isLeaf := tree.left == nil || tree.right == nil
	if isLeaf {
		return r.executeTerminalNode(tree, level)
	}

	left, err := codegenTree(r, tree.left, level+1)
	if err != nil {
		return "", err
	}
	right, err := codegenTree(r, tree.right, level+1)
	if err != nil {
		return "", err
	}

	return r.executeDecisionNode(tree, level, left, right)
}

// GenerateFile generates a .go file containing a function that implements the XGB model.
func GenerateFile(
	inputJSON string,
	packageName,
	funcName,
	outputFile string,
) error {
	x, err := readModel(inputJSON)
	if err != nil {
		return err
	}

	trees, err := readTrees(x)
	if err != nil {
		return err
	}

	r, err := newRenderer()
	if err != nil {
		return err
	}

	numClasses, err := x.Learner.ModelParams.NumClass.Int64()
	if err != nil {
		return fmt.Errorf("error parsing num_class as int: %w", err)
	}

	baseScore, err := x.Learner.ModelParams.BaseScore.Float64()
	if err != nil {
		return fmt.Errorf("error parsing base_score as float: %w", err)
	}

	code, err := generateSource(packageName, funcName, float32(baseScore), numClasses, trees, r)
	if err != nil {
		return err
	}

	if err := os.WriteFile(outputFile, []byte(code), 0o644); err != nil {
		return fmt.Errorf("error writing file: %w", err)
	}
	return nil
}
