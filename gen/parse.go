package gen

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type xgbModel struct {
	Learner struct {
		Attributes struct {
			BestIteration json.Number `json:"best_iteration"`
		} `json:"attributes"`
		GradientBooster struct {
			Model struct {
				TreeInfo []int     `json:"tree_info"`
				Trees    []xgbTree `json:"trees"`
			} `json:"model"`
		} `json:"gradient_booster"`
		ModelParams struct {
			BaseScore  json.Number `json:"base_score"`
			NumClass   json.Number `json:"num_class"`
			NumFeature json.Number `json:"num_feature"`
		} `json:"learner_model_param"`
	} `json:"learner"`
}

type xgbTree struct {
	DefaultLeft     []int     `json:"default_left"`
	LeftChildren    []int     `json:"left_children"`
	RightChildren   []int     `json:"right_children"`
	SplitConditions []float64 `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	TreeParam       struct {
		NumNodes json.Number `json:"num_nodes"`
	} `json:"tree_param"`
}

func readModel(inputJSON string) (*xgbModel, error) {
	fh, err := os.Open(filepath.Clean(inputJSON))
	if err != nil {
		return nil, fmt.Errorf("error opening file: %w", err)
	}
	defer fh.Close()

	var x xgbModel
	if err := json.NewDecoder(fh).Decode(&x); err != nil {
		return nil, fmt.Errorf("error decoding JSON: %w", err)
	}

	return &x, nil
}

func readTrees(x *xgbModel) ([]*node, error) {
	var trees []*node
	for i := range x.Learner.GradientBooster.Model.Trees {
		t := x.Learner.GradientBooster.Model.Trees[i]

		treeInfo, err := parseTreeInfo(t)
		if err != nil {
			return nil, err
		}

		treeInfo.data.Class = x.Learner.GradientBooster.Model.TreeInfo[i]
		trees = append(trees, treeInfo)
	}

	if x.Learner.Attributes.BestIteration == "" {
		// model was not trained with early stopping, use all the trees
		return trees, nil
	}

	bestIteration, err := x.Learner.Attributes.BestIteration.Int64()
	if err != nil {
		return nil, fmt.Errorf(
			"error parsing best_iteration as int: %w",
			err,
		)
	}

	return trees[:bestIteration+1], nil
}

type node struct {
	left  *node
	right *node
	data  nodeData
}

type nodeData struct {
	Class          int
	DefaultLeft    int
	ID             int
	SplitCondition float64
	SplitIndex     int
}

func parseTreeInfo(xt xgbTree) (*node, error) {
	numNodes, err := xt.TreeParam.NumNodes.Int64()
	if err != nil {
		return nil, fmt.Errorf(
			"error parsing num_nodes as an integer: %w",
			err,
		)
	}

	nodes := make([]node, numNodes)
	for i := 0; i < int(numNodes); i++ {
		nodes[i].data = nodeData{
			DefaultLeft:    xt.DefaultLeft[i],
			ID:             i,
			SplitCondition: xt.SplitConditions[i],
			SplitIndex:     xt.SplitIndices[i],
		}

		left := xt.LeftChildren[i]
		right := xt.RightChildren[i]

		if left == -1 { // No child
			continue
		}

		nodes[i].left = &nodes[left]
		nodes[i].right = &nodes[right]
	}

	return &nodes[0], nil // Root node
}
