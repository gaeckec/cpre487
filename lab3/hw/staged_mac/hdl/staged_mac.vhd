-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- staged_mac.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a basic staged axi-stream mac unit. It
-- multiplies two integer/Q values togeather and accumulates them.
--
-- NOTES:
-- 10/25/21 by MPD::Inital template creation
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity staged_mac is
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

end staged_mac;


architecture behavioral of staged_mac is
    -- Internal Signals
    constant INTEGER_BITS         : integer := C_DATA_WIDTH/2;
    constant FIXED_BITS           : integer := C_DATA_WIDTH/2;
    subtype fixed_t is signed(INTEGER_BITS+FIXED_BITS-1 downto 0);
    constant fixed_t_zero : fixed_t := (others => '0');    
--    constant fixed_t_one  : fixed_t  := x"00010000";
    
    
    signal input_one, input_two, mult_bits : fixed_t;
    signal sum_bits     : fixed_t := fixed_t_zero;
    
    signal t_last, t_last_MV : std_logic;
	
	-- Mac state
    type STATE_TYPE is (WAIT_FOR_VALUES, MULT_VALUE, ADD_VALUES, OUTPUT_DATA);
    signal state : STATE_TYPE;
    
    
	
	-- Debug signals, make sure we aren't going crazy
    signal mac_debug : std_logic_vector(31 downto 0);

begin

    -- Interface signals


    -- Internal signals
	
	
	-- Debug Signals
    mac_debug <= x"00000000";  -- Double checking sanity
   
   process (ACLK) is
   begin 
    if rising_edge(ACLK) then  -- Rising Edge

      -- Reset values if reset is low
      if ARESETN = '0' then  -- Reset
        state       <= WAIT_FOR_VALUES;
        sum_bits    <= fixed_t_zero;

      else
        case state is  -- State
            -- Wait here until we receive values
            when WAIT_FOR_VALUES =>
                -- Wait here until we recieve valid values
			    SD_AXIS_TREADY <= '1';
			    if SD_AXIS_TVALID = '1' then
			        input_one <= signed(SD_AXIS_TDATA(C_DATA_WIDTH-1 downto 0));
			        input_two <= signed(SD_AXIS_TDATA(2*C_DATA_WIDTH-1 downto C_DATA_WIDTH));
			        SD_AXIS_TREADY <= '0';
			        t_last <= SD_AXIS_TLAST;
			        state <= MULT_VALUE;
			    end if;
			
			when MULT_VALUE => 
			    mult_bits <= input_one * input_two;
			    t_last_MV <= t_last;
			    state <= ADD_VALUES;
			    
			
			when ADD_VALUES => 
			    sum_bits <= sum_bits + mult_bits;
			    
			    if t_last_MV = '1' then
			        state <= OUTPUT_DATA;
			    else
			        state <= WAIT_FOR_VALUES;
			    end if;
			    
			when OUTPUT_DATA =>     
			    MO_AXIS_TVALID <= '1';
			    MO_AXIS_TDATA <= std_logic_vector(sum_bits);
			    if MO_AXIS_TREADY = '1' then
			        MO_AXIS_TVALID <= '0';
			        sum_bits <= fixed_t_zero;
			        state <= WAIT_FOR_VALUES;
			    end if;
			
			
            when others =>
                state <= WAIT_FOR_VALUES;
                -- Not really important, this case should never happen
                -- Needed for proper synthisis         
        end case;  -- State
      end if;  -- Reset

    end if;  -- Rising Edge
   end process;
end architecture behavioral;