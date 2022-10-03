-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- piped_mac.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a basic piplined axi-stream mac unit. It
-- multiplies two integer/Q values togeather and accumulates them.
--
-- NOTES:
-- 10/25/21 by MPD::Inital template creation
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity piped_mac is
  generic(
      -- Parameters of mac
      C_DATA_WIDTH : integer := 32
    );
	port (
        ACLK	: in	std_logic;
		ARESETN	: in	std_logic;       

        -- AXIS slave data interface
		SD_AXIS_TREADY	: out	std_logic;
		SD_AXIS_TDATA	: in	std_logic_vector(C_DATA_WIDTH*2-1 downto 0);  -- Packed data input
		SD_AXIS_TLAST	: in	std_logic;
        SD_AXIS_TUSER   : in    std_logic;  -- Should we treat this first value in the stream as an inital accumulate value?
		SD_AXIS_TVALID	: in	std_logic;
        SD_AXIS_TID     : in    std_logic_vector(7 downto 0);

        -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	: out	std_logic;
		MO_AXIS_TDATA	: out	std_logic_vector(C_DATA_WIDTH-1 downto 0);
		MO_AXIS_TLAST	: out	std_logic;
		MO_AXIS_TREADY	: in	std_logic;
		MO_AXIS_TID     : out   std_logic_vector(7 downto 0)
    );

attribute SIGIS : string; 
attribute SIGIS of ACLK : signal is "Clk"; 

end piped_mac;


architecture behavioral of piped_mac is
    -- Internal Signals
    constant INTEGER_BITS : integer := C_DATA_WIDTH/2;
    constant FIXED_BITS   : integer := C_DATA_WIDTH/2;
    
    subtype fixed_t is signed(INTEGER_BITS+FIXED_BITS-1 downto 0);
    subtype fixed_t_wide is signed((INTEGER_BITS+FIXED_BITS)*2-1 downto 0);
    subtype fixed_t_wide_check is signed((INTEGER_BITS+FIXED_BITS)*2 downto 0);
    
    constant fixed_t_zero : fixed_t := (others => '0');
    constant fixed_t_wide_zero : fixed_t_wide := (others => '0');
    
--    signal input_0       : fixed_t := fixed_t_zero;
--    signal input_1       : fixed_t := fixed_t_zero;
    signal mult_bits     : fixed_t_wide := fixed_t_wide_zero;
    signal sum_bits      : fixed_t_wide_check := (others => '0');
    
    signal t_last_0, t_last_1  : std_logic;
	signal prev_TID_0, prev_TID_1  : std_logic_vector(7 downto 0);
	signal double_op, first_out : std_logic;
	
	-- Mac stages
    type PIPE_STAGES is (WAIT_FOR_VALUES, ADD, OUT_1, OUT_2);

	
	-- Debug signals, make sure we aren't going crazy
    signal mac_debug : std_logic_vector(31 downto 0);

begin

    -- Interface signals

    -- Internal signal
    --t_last_0 <= SD_AXIS_TLAST;

    
	-- Debug Signals
   mac_debug <= x"00000002";  -- Double checking sanity
   
   process (ACLK) is
   begin 
    if rising_edge(ACLK) then  -- Rising Edge

      -- Reset values if reset is low
      if ARESETN = '0' then  -- Reset
        mult_bits <= (others => '0');
        sum_bits  <= (others => '0');
        
        t_last_0 <= '0';
        t_last_1 <= '0';
        
        prev_TID_0 <= (others => '0');
        prev_TID_1 <= (others => '0');
        
        double_op <= '0';
        first_out <= '0';
        
		
      else
        for i in PIPE_STAGES'left to PIPE_STAGES'right loop
            case i is  -- Stages
                when WAIT_FOR_VALUES =>
                    if SD_AXIS_TVALID = '1' then
                        if SD_AXIS_TUSER = '1' then
                            mult_bits <= signed(resize(signed(SD_AXIS_TDATA(C_DATA_WIDTH-1 downto 0)), mult_bits'length));
                        else
                            mult_bits <= signed(SD_AXIS_TDATA(C_DATA_WIDTH-1 downto 0)) * signed(SD_AXIS_TDATA(C_DATA_WIDTH*2-1 downto C_DATA_WIDTH));
                        end if;
                        
                        t_last_0  <= SD_AXIS_TLAST;
                        t_last_1  <= '0';
                        prev_TID_0 <= SD_AXIS_TID; 
                    
                    end if;
                    
                when ADD =>
                    -- Accumulate until last value in stream
                    if t_last_1 = '0' then
                        sum_bits <= sum_bits + mult_bits;
                        t_last_1 <= t_last_0;
                        t_last_0 <= SD_AXIS_TLAST;
                        prev_TID_1 <= prev_TID_0;
                                            
                    end if;               
                    
                    if SD_AXIS_TLAST = '1' and t_last_0 = '1' and (SD_AXIS_TID /= prev_TID_0) then
                        double_op <= '1';
                    
                    end if;            
                                                        
                when OUT_1 =>
                    if t_last_1 = '1' then
                        first_out <= '1';
                            
                        -- Logic for truncation of summation
                        if(sum_bits(2*C_DATA_WIDTH downto INTEGER_BITS+FIXED_BITS+FIXED_BITS) > 0) then
                            MO_AXIS_TDATA(C_DATA_WIDTH-1) <= '0';
                            MO_AXIS_TDATA(C_DATA_WIDTH-2 downto 0) <= (others => '1');
                            
                        elsif(sum_bits(2*C_DATA_WIDTH downto INTEGER_BITS+FIXED_BITS+FIXED_BITS) < -1) then
                            MO_AXIS_TDATA(C_DATA_WIDTH-1) <= '1';
                            MO_AXIS_TDATA(C_DATA_WIDTH-2 downto 0) <= (others => '0');
                            
                        else 
                            MO_AXIS_TDATA <= std_logic_vector(sum_bits(INTEGER_BITS+FIXED_BITS+FIXED_BITS-1 downto FIXED_BITS));
                        
                        end if;
                        
                        -- AXI Bus handshake
                        MO_AXIS_TVALID <= '1';
                        if MO_AXIS_TREADY = '1' then
                            MO_AXIS_TDATA <= std_logic_vector(sum_bits(C_DATA_WIDTH-1 downto 0));
                            MO_AXIS_TVALID <= '0';
                            MO_AXIS_TLAST <= t_last_1;
                            MO_AXIS_TID <= prev_TID_1;
                            t_last_0 <= '0';
                            
                        end if;
                    end if;
            
                when OUT_2 =>
                    if double_op = '1' and first_out = '1' then
                        double_op <= '0';
                        first_out <= '0';
                    
                        if(mult_bits(2*C_DATA_WIDTH-1 downto INTEGER_BITS+FIXED_BITS+FIXED_BITS) > 0) then
                            MO_AXIS_TDATA(C_DATA_WIDTH-1) <= '0';
                            MO_AXIS_TDATA(C_DATA_WIDTH-2 downto 0) <= (others => '1');
                            
                        elsif(mult_bits(2*C_DATA_WIDTH-1 downto INTEGER_BITS+FIXED_BITS+FIXED_BITS) < -1) then
                            MO_AXIS_TDATA(C_DATA_WIDTH-1) <= '1';
                            MO_AXIS_TDATA(C_DATA_WIDTH-2 downto 0) <= (others => '0');
                            
                        else 
                            MO_AXIS_TDATA <= std_logic_vector(mult_bits(INTEGER_BITS+FIXED_BITS+FIXED_BITS-1 downto FIXED_BITS));
                        
                        end if;
                        
                        -- AXI Bus handshake
                        MO_AXIS_TVALID <= '1';
                        if MO_AXIS_TREADY = '1' then
                            MO_AXIS_TDATA <= std_logic_vector(mult_bits(C_DATA_WIDTH-1 downto 0));
                            MO_AXIS_TVALID <= '0';
                            MO_AXIS_TLAST <= t_last_1;
                            MO_AXIS_TID <= prev_TID_0;
                            t_last_1 <= '0';
                            
                        end if;                
                    end if; 

            end case;  -- Stages
		end loop;  -- Stages
      end if;  -- Reset

    end if;  -- Rising Edge
   end process;
end architecture behavioral;
