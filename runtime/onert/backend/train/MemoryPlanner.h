



/**
 * @brief Structure to have memory offset and size
 */
struct Block
{
  uint32_t offset;
  size_t size;
};

/**
 * @brief Interface to plan memory
 */
struct IMemoryPlanner
{
  using MemoryPlans = ir::OperandIndexMap<Block>;

  /**
   * @brief Claim memory for operand
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  virtual void claim(const ir::OperandIndex &, size_t) = 0;
  /**
   * @brief Release memory for operand
   * @param[in] index The operand index
   */
  virtual void release(const ir::OperandIndex &) = 0;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  virtual uint32_t capacity() = 0;
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  virtual MemoryPlans &memory_plans() = 0;

  virtual ~IMemoryPlanner() = default;
};
