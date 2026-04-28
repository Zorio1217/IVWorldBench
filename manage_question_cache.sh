#!/bin/bash
"""
问题缓存管理脚本
提供便捷的缓存管理功能
"""

CACHE_MANAGER="question_cache_manager.py"
QA_GENERATOR="adaptive_qa_generator.py"

# 显示使用帮助
show_help() {
    echo "问题缓存管理脚本"
    echo "用法:"
    echo "  $0 stats [dimension]          - 显示缓存统计信息"
    echo "  $0 clear [dimension]          - 清除缓存"
    echo "  $0 migrate <file> <dimension> - 迁移现有问题到缓存"
    echo ""
    echo "示例:"
    echo "  $0 stats                                    # 显示所有维度的缓存统计"
    echo "  $0 stats dynamic_attribute                  # 显示特定维度的缓存统计"
    echo "  $0 clear                                    # 清除所有缓存"
    echo "  $0 clear dynamic_attribute                  # 清除特定维度的缓存"
    echo "  $0 migrate file.json dynamic_attribute      # 迁移问题到缓存"
}

# 显示缓存统计
show_stats() {
    if [ -z "$1" ]; then
        echo "显示所有维度的缓存统计:"
        python "$CACHE_MANAGER" --action stats
    else
        echo "显示维度 '$1' 的缓存统计:"
        python "$CACHE_MANAGER" --action stats --dimension "$1"
    fi
}

# 清除缓存
clear_cache() {
    if [ -z "$1" ]; then
        echo "清除所有缓存..."
        python "$CACHE_MANAGER" --action clear
    else
        echo "清除维度 '$1' 的缓存..."
        python "$CACHE_MANAGER" --action clear --dimension "$1"
    fi
}

# 迁移问题到缓存
migrate_questions() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "错误: 迁移需要指定文件路径和维度"
        echo "用法: $0 migrate <file> <dimension>"
        exit 1
    fi
    
    if [ ! -f "$1" ]; then
        echo "错误: 文件 '$1' 不存在"
        exit 1
    fi
    
    echo "迁移文件 '$1' 中的问题到维度 '$2' 的缓存..."
    python "$QA_GENERATOR" --json_file "$1" --dimension "$2" --migrate_cache
}

# 主逻辑
case "$1" in
    "stats")
        show_stats "$2"
        ;;
    "clear")
        clear_cache "$2"
        ;;
    "migrate")
        migrate_questions "$2" "$3"
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "错误: 未知操作 '$1'"
        show_help
        exit 1
        ;;
esac
