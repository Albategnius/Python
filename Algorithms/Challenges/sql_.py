/* write your SQL query below */

SELECT m.GroupID, c.CompanyName, Count(*) as Count
FROM maintable_PNVZK m
Inner join cb_vendor1information1 c on m.GroupID = c.GroupID
Group by m.GroupID, c.CompanyName
order by Count, m.GroupID asc



SELECT t.ID, t.Name, c.DivisionName,m.Name as ManagerName,
t.Salary  
FROM maintable_PVDW3 t
Inner join
cb_company1divisions1 as c 
on c.ID = t.DivisionID
Inner join maintable_PVDW3 m on m.ID = t.ManagerID
order by Salary
Desc Limit 1
Offset 1


SELECT FirstName,LastName,ReportsTo, Position,Age,
  case when ReportsTo = 'Jenny Richards' then 'CEO'
  else 'None' end
  as "Boss Title"
FROM 11maintable_E4YPE
where ReportsTo ='Jenny Richards' or ReportsTo is null
order by age



SELECT count(*) FROM 11maintable_2KDZI
where FirstName like '%e%'
and char_length(LastName) > 5



SELECT ReportsTo, count(*) as Members,round(avg(Age)) as 'Average Age' 
FROM 11maintable_IEHSI
where ReportsTo is not null
Group by ReportsTo 



SELECT monthname(t.DateJoined) as Month,
  count(t.DateJoined)  - 
  (
    select count(t1.DateJoined)
    from maintable_10EVQ t1
    where Month(t1.DateJoined) = Month(t.DateJoined) - 1
  ) as 11MonthToMonthChange
FROM maintable_10EVQ t
where Month(t.DateJoined) != 1
Group by monthname(t.DateJoined)
order by t.DateJoined asc



SELECT 
t1.GroupID,FirstName,
LastName,Job, ExternalID,CompanyName,
  count(*) as count  
FROM maintable_FE50M t
inner join cb_vendor1information1 t1
on t.GroupID = t1.GroupID
Group by FirstName
order by 7,6 
